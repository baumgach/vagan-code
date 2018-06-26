# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from skimage import transform  # used for CAM saliency

from tfwrapper import losses
from tfwrapper import utils as tf_utils
import config.system as sys_config
from grad_accum_optimizers import grad_accum_optimizer_classifier


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


class classifier:

    def __init__(self, exp_config, data, fixed_batch_size=None):

        self.exp_config = exp_config
        self.data = data

        self.nlabels = exp_config.nlabels

        self.image_tensor_shape = [fixed_batch_size] + list(exp_config.image_size) + [1]
        self.labels_tensor_shape = [fixed_batch_size]

        self.x_pl = tf.placeholder(tf.float32, shape=self.image_tensor_shape, name='images')
        self.y_pl = tf.placeholder(tf.uint8, shape=self.labels_tensor_shape, name='labels')

        self.lr_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.training_pl = tf.placeholder(tf.bool, shape=[], name='training_time')

        self.l_pl_ = exp_config.classifier_net(self.x_pl, nlabels=self.nlabels, training=self.training_pl)
        self.y_pl_ = tf.nn.softmax(self.l_pl_)
        self.p_pl_ = tf.argmax(self.y_pl_, axis=-1)

        # Add to the Graph the Ops for loss calculation.

        self.classifier_loss = self.classification_loss()
        self.weights_norm = self.weight_norm()

        self.total_loss = self.classifier_loss + self.weights_norm

        self.global_step = tf.train.get_or_create_global_step()  # Used in batch renormalisation

        self.opt = grad_accum_optimizer_classifier(loss=self.total_loss,
                                                   optimizer=self._get_optimizer(),
                                                   variable_list=tf.trainable_variables(),
                                                   n_accum=exp_config.n_accum_grads)

        self.global_step = tf.train.get_or_create_global_step()
        self.increase_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        self.saver_best_xent = tf.train.Saver(max_to_keep=2)
        self.saver_best_f1 = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)

    def classification_loss(self):

        y_for_loss = tf.one_hot(self.y_pl, depth=self.nlabels)

        classification_loss = losses.cross_entropy_loss(logits=self.l_pl_,
                                                        labels=y_for_loss)

        return classification_loss

    def weight_norm(self):

        weights_norm = tf.reduce_sum(
            input_tensor=self.exp_config.weight_decay * tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

        return weights_norm


    def train(self):

        # Sort out proper logging
        self._setup_log_dir_and_continue_mode()

        # Create tensorboard summaries
        self._make_tensorboard_summaries()

        self.curr_lr = self.exp_config.learning_rate
        schedule_lr = True if self.exp_config.divide_lr_frequency is not None else False

        logging.info('===== RUNNING EXPERIMENT ========')
        logging.info(self.exp_config.experiment_name)
        logging.info('=================================')

        # initialise all weights etc..
        self.sess.run(tf.global_variables_initializer())

        # Restore session if there is one
        if self.continue_run:
            self.saver.restore(self.sess, self.init_checkpoint_path)

        logging.info('Starting training:')

        best_val = np.inf
        best_diag_f1_score = 0

        for step in range(self.init_step, self.exp_config.max_iterations):

            # If learning rate is scheduled
            if self.exp_config.warmup_training:
                if step < 50:
                    self.curr_lr = self.exp_config.learning_rate / 10.0
                elif step == 50:
                    self.curr_lr = self.exp_config.learning_rate

            if schedule_lr and step > 0 and step % self.exp_config.divide_lr_frequency == 0:
                self.curr_lr /= 10.0
                logging.info('Updating learning rate to: %f' % self.curr_lr)

            batch_x_dims = [self.exp_config.batch_size] + list(self.exp_config.image_size) + [1]
            batch_y_dims = [self.exp_config.batch_size]
            feed_dict = {self.x_pl: np.zeros(batch_x_dims),  # dummy variables will be replaced in optimizer
                         self.y_pl: np.zeros(batch_y_dims),
                         self.training_pl: True,
                         self.lr_pl: self.curr_lr}

            start_time = time.time()
            loss_value = self.opt.do_training_step(sess=self.sess,
                                                   sampler=self.data.train.next_batch,
                                                   batch_size=self.exp_config.batch_size,
                                                   feed_dict=feed_dict,
                                                   img_pl=self.x_pl,
                                                   lbl_pl=self.y_pl,
                                                   loss=self.total_loss)
            elapsed_time = time.time() - start_time


            ###  Tensorboard updates, Model Saving, and Validation

            # Update tensorboard
            if step % 5 == 0:

                logging.info('Step %d: loss = %.2f (One update step took %.3f sec)' % (step, loss_value, elapsed_time))

                summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            # Do training evaluation
            if (step + 1) % self.exp_config.train_eval_frequency == 0:

                # Evaluate against the training set
                logging.info('Training Data Eval:')
                [train_loss, train_diag_f1] = self._do_validation(self.data.train.iterate_batches)

                train_summary_msg = self.sess.run(self.train_summary,
                                                  feed_dict={self.train_error_: train_loss,
                                                             self.train_diag_f1_score_: train_diag_f1}
                                             )
                self.summary_writer.add_summary(train_summary_msg, step)


            # Do validation set evaluation
            if (step + 1) % self.exp_config.val_eval_frequency == 0:

                checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=step)

                # Evaluate against the validation set.
                logging.info('Validation Data Eval:')

                [val_loss, val_diag_f1] = self._do_validation(self.data.validation.iterate_batches)

                val_summary_msg = self.sess.run(self.val_summary,
                                                feed_dict={self.val_error_: val_loss,
                                                           self.val_diag_f1_score_: val_diag_f1}
                                           )
                self.summary_writer.add_summary(val_summary_msg, step)

                if val_diag_f1 >= best_diag_f1_score:
                    best_diag_f1_score = val_diag_f1
                    best_file = os.path.join(self.log_dir, 'model_best_f1.ckpt')
                    self.saver_best_f1.save(self.sess, best_file, global_step=step)
                    logging.info( 'Found new best F1 score on validation set! - %f -  Saving model_best_f1.ckpt' % val_diag_f1)

                if val_loss < best_val:
                    best_val = val_loss
                    best_file = os.path.join(self.log_dir, 'model_best_xent.ckpt')
                    self.saver_best_xent.save(self.sess, best_file, global_step=step)
                    logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

            self.sess.run(self.increase_global_step)


    def load_weights(self, log_dir=None, type='latest', **kwargs):

        if not log_dir:
            log_dir = self.log_dir

        if type=='latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type=='best_f1':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_f1.ckpt')
        elif type=='best_xent':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_xent.ckpt')
        elif type=='iter':
            assert 'iteration' in kwargs, "argument 'iteration' must be provided for type='iter'"
            iteration = kwargs['iteration']
            init_checkpoint_path = os.path.join(log_dir, 'model.ckpt-%d' % iteration)
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_wasserstein/iter.' % type)

        self.saver.restore(self.sess, init_checkpoint_path)


    def predict(self, images):

        prediction, softmax = self.sess.run([self.p_pl_, self.y_pl_],
                                            feed_dict={self.x_pl: images, self.training_pl: False})

        confidence = softmax[prediction]

        return prediction, confidence


    def initialise_saliency(self, mode, **kwargs):

        # self.saliency = tf.gradients(self.l_pl_, self.x_pl)

        self.saliency_mode = mode
        self.lbl_selector = tf.placeholder(tf.int32, name='lbl_selector')

        if mode == 'guided_backprop':

            g = tf.get_default_graph()
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                l_sal = self.exp_config.classifier_net(self.x_pl, nlabels=self.nlabels, training=self.training_pl, scope_reuse=True)
                y_sal = tf.nn.softmax(l_sal)[0,self.lbl_selector]
            self.saliency = tf.gradients(y_sal, self.x_pl)

        elif mode in ['backprop', 'integrated_gradients']:

            # self.saliency = tf.gradients(self.l_pl_[:,self.lbl_selector], self.x_pl)
            self.saliency = tf.gradients(self.y_pl_[:,self.lbl_selector], self.x_pl)[0]
            # self.saliency = tf.gradients(self.l_pl_, self.x_pl)

        elif mode == 'additive_pertubation':

            beta = 2.0 if not 'beta' in kwargs else kwargs['beta']
            l1 = 1e-2 if not 'l1' in kwargs else kwargs['l1']
            l2 = 1e-1 if not 'l2' in kwargs else kwargs['l2']
            sal_lr = 0.1 if not 'sal_lr' in kwargs else kwargs['sal_lr']

            # beta = 2.0 if not 'beta' in kwargs else kwargs['beta']
            # l1 = 1e-2 if not 'l1' in kwargs else kwargs['l1']
            # l2 = 1e-5 if not 'l2' in kwargs else kwargs['l2']
            # sal_lr = 0.01 if not 'sal_lr' in kwargs else kwargs['sal_lr']

            with tf.variable_scope('addpert_saliency'):  # So we know what to initialize later
                self.sal_mask_var = tf.Variable(tf.constant(0.0, shape=self.image_tensor_shape, dtype=tf.float32), name='perturbation_mask', trainable=True)

            network_input = self.x_pl + self.sal_mask_var
            logits = self.exp_config.classifier_net(network_input, nlabels=self.nlabels, training=self.training_pl, scope_reuse=True)
            y = tf.nn.softmax(logits)[0, self.lbl_selector]

            l1_norm = tf.reduce_sum(tf.abs(self.sal_mask_var))
            total_variation = tf_utils.total_variation(self.sal_mask_var, beta=beta)
            loss = y + l1*l1_norm + l2*total_variation

            with tf.variable_scope('addpert_saliency'):  # This is to avoid double definition of ADAM variables
                sal_optimiser = tf.train.AdamOptimizer(learning_rate=sal_lr)
                self.sal_train_op = sal_optimiser.minimize(loss, var_list=self.sal_mask_var)

        elif mode == 'CAM':

            graph = tf.get_default_graph()

            try:
                self.sal_feature_maps = graph.get_tensor_by_name('classifier/feature_maps/Relu:0')
                self.sal_weight_layer = [v for v in tf.global_variables() if 'weight_layer' in v.name][0]
            except KeyError:
                raise KeyError("The weights or feature maps required for CAM could not be found! "
                                "Make sure you use a network designed for CAM. ")

        else:

            raise ValueError('Unknown saliency mode')

        # self.saver = tf.train.Saver()

    def compute_saliency(self, image, label, **kwargs):

        if self.saliency_mode in ['backprop', 'guided_backprop']:
            return self.sess.run(self.saliency, feed_dict={self.x_pl: image, self.lbl_selector: label, self.training_pl: False})

        elif self.saliency_mode == 'additive_pertubation':

            sal_var_initializers = [v.initializer for v in tf.global_variables() if v.name.startswith("addpert_saliency")]
            self.sess.run(sal_var_initializers)

            # num_iter = 100 if not 'num_iter' in kwargs else kwargs['num_iter']
            num_iter = 100 if not 'num_iter' in kwargs else kwargs['num_iter']
            for iter in range(num_iter):
                self.sess.run(self.sal_train_op, feed_dict={self.x_pl: image, self.lbl_selector: label, self.training_pl: False})
            return -self.sess.run(self.sal_mask_var)

        elif self.saliency_mode == 'integrated_gradients':
            base_image = np.zeros(self.image_tensor_shape)
            mask = np.zeros(self.image_tensor_shape)
            m = 100 if not 'num_steps' in kwargs else kwargs['num_steps']
            for k in range(m):
                curr_img = base_image + (float(k)/m) * (image - base_image)
                mask += self.sess.run(self.saliency, feed_dict={self.x_pl: curr_img, self.lbl_selector: label, self.training_pl: False})

            return np.divide((image - base_image), m) * mask


        elif self.saliency_mode == 'CAM':

            feature_maps_eval, weights_eval = self.sess.run([self.sal_feature_maps, self.sal_weight_layer],
                                                            feed_dict={self.x_pl: image,  self.training_pl: False})

            feature_maps_eval = np.squeeze(feature_maps_eval)
            weights_eval = np.squeeze(weights_eval)

            K = weights_eval.shape[0]
            mask = np.zeros(feature_maps_eval.shape[:2])

            for kk in range(K):
                w_k = weights_eval[kk, label]
                mask += w_k * np.squeeze(feature_maps_eval[..., kk])

            return transform.resize(mask, np.squeeze(image).shape, order=1, preserve_range=True, mode='constant')

        else:
            raise ValueError('Saliency mode unknown or not properly set')
        # return self.sess.run(self.saliency, feed_dict={self.x_pl: image, self.training_pl: False})

    ### HELPER FUNCTIONS ###################################################################################

    def _make_tensorboard_summaries(self):

        tf.summary.scalar('learning_rate', self.lr_pl)

        tf.summary.scalar('classifier_loss', self.classifier_loss)
        tf.summary.scalar('weights_norm', self.weights_norm)
        tf.summary.scalar('total_loss', self.total_loss)

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        self.val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error_diag')
        val_error_summary = tf.summary.scalar('validation_loss', self.val_error_)

        self.val_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='val_diag_f1')
        val_f1_diag_summary = tf.summary.scalar('validation_diag_f1', self.val_diag_f1_score_)

        self.val_summary = tf.summary.merge([val_error_summary, val_f1_diag_summary])

        self.train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error_diag')
        train_error_summary = tf.summary.scalar('training_loss', self.train_error_)

        self.train_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='train_diag_f1')
        train_diag_f1_summary = tf.summary.scalar('training_diag_f1', self.train_diag_f1_score_)


        self.train_summary = tf.summary.merge([train_error_summary, train_diag_f1_summary])


    def _get_optimizer(self):

        if self.exp_config.optimizer_handle == tf.train.AdamOptimizer:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    beta1=self.exp_config.beta1,
                                                    beta2=self.exp_config.beta2)
        if self.exp_config.momentum is not None:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    momentum=self.exp_config.momentum)
        else:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl)


    def _setup_log_dir_and_continue_mode(self):

        # Default values
        self.log_dir = os.path.join(sys_config.log_root, 'classifier', self.exp_config.experiment_name)
        self.init_checkpoint_path = None
        self.continue_run = False
        self.init_step = 0

        # If a checkpoint file already exists enable continue mode
        if tf.gfile.Exists(self.log_dir):
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(self.log_dir, 'model.ckpt')
            if init_checkpoint_path is not False:

                self.init_checkpoint_path = init_checkpoint_path
                self.continue_run = True
                self.init_step = int(self.init_checkpoint_path.split('/')[-1].split('-')[-1])
                self.log_dir += '_cont'

                logging.info('--------------------------- Continuing previous run --------------------------------')
                logging.info('Checkpoint path: %s' % self.init_checkpoint_path)
                logging.info('Latest step was: %d' % self.init_step)
                logging.info('------------------------------------------------------------------------------------')

        tf.gfile.MakeDirs(self.log_dir)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Copy experiment config file to log_dir for future reference
        shutil.copy(self.exp_config.__file__, self.log_dir)


    def _eval_predict(self, images, labels):

        prediction, loss = self.sess.run([self.p_pl_, self.total_loss],
                                         feed_dict={self.x_pl: images,
                                                    self.y_pl: labels,
                                                    self.training_pl: False})

        return prediction, loss


    def _do_validation(self, iterator):

        diag_loss_ii = 0
        num_batches = 0
        predictions_diag = []
        predictions_diag_gt = []

        for batch in iterator(self.exp_config.batch_size):

            x, y = batch

            # Skip incomplete batches
            if y.shape[0] < self.exp_config.batch_size:
                continue

            c_d_preds, c_d_loss = self._eval_predict(x, y)

            # This converts the labels back into the original format. I.e. [0,1,1,0] will become [0,2,2,0] again if
            # 1 didn't exist in the dataset.
            # c_d_preds = [exp_config.label_list[int(pp)] for pp in c_d_preds]
            # y_gts = [exp_config.label_list[pp] for pp in y]

            num_batches += 1
            predictions_diag += list(c_d_preds)
            diag_loss_ii += c_d_loss
            predictions_diag_gt += list(y)

        avg_loss = (diag_loss_ii / num_batches)

        average_mode = 'binary' if self.nlabels == 2 else 'micro'
        f1_diag_score = f1_score(np.asarray(predictions_diag_gt), np.asarray(predictions_diag), average=average_mode)

        logging.info('  Average loss: %0.04f, diag f1_score: %0.04f' % (avg_loss, f1_diag_score, ))

        return avg_loss, f1_diag_score


