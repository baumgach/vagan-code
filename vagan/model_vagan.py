# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import logging
import shutil
import numpy as np
import os.path
import tensorflow as tf

import config.system as sys_config
from grad_accum_optimizers import grad_accum_optimizer_gan
from tfwrapper import utils as tf_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

class vagan:

    """
    This class contains all the methods for defining training and evaluating the VA-GAN method
    """

    def __init__(self, exp_config, data, fixed_batch_size=None):

        """
        Initialise the VA-GAN model with the two required networks, loss functions, etc... 
        :param exp_config: An experiment config file
        :param data: A handle to the data object that should be used
        :param fixed_batch_size: Optionally, a fixed batch size can be selected. If None, the batch_size will stay
                                 flexible. 
        """

        self.exp_config = exp_config
        self.data = data

        self.mode3D = True if len(exp_config.image_size) == 3 else False

        self.critic_net = exp_config.critic_net
        self.generator_net = exp_config.generator_net

        self.sampler_c1 = lambda bs: data.trainAD.next_batch(bs)[0]
        self.sampler_c0 = lambda bs: data.trainCN.next_batch(bs)[0]

        self.img_tensor_shape = [fixed_batch_size] + list(exp_config.image_size) + [1]
        self.batch_size = exp_config.batch_size

        # Generate placeholders for the images and labels.
        self.training_pl_cri = tf.placeholder(tf.bool, name='training_phase_critic')
        self.training_pl_gen = tf.placeholder(tf.bool, name='training_phase_generator')

        self.lr_pl = tf.placeholder(tf.float32, name='learning_rate')

        self.x_c0 = tf.placeholder(tf.float32, self.img_tensor_shape, name='c0_img')
        self.x_c1 = tf.placeholder(tf.float32, self.img_tensor_shape, name='c1_img')

        # network outputs
        self.M = self.generator_net(self.x_c1, self.training_pl_gen)
        self.y_c0_ = self.x_c1 + self.M

        if exp_config.use_tanh:
            self.y_c0_ = tf.tanh(self.y_c0_)

        self.D = self.critic_net(self.x_c0, self.training_pl_cri, scope_reuse=False)
        self.D_ = self.critic_net(self.y_c0_, self.training_pl_cri, scope_reuse=True)

        # Generator and critic losses
        self.gen_loss = self.generator_loss()
        self.cri_loss = self.critic_loss()

        # Make optimizers
        train_vars = tf.trainable_variables()
        self.gen_vars = [v for v in train_vars if v.name.startswith("generator")]
        self.cri_vars = [v for v in train_vars if v.name.startswith("critic")]

        self.gen_opt = grad_accum_optimizer_gan(loss=self.gen_loss,
                                                optimizer=self._get_optimizer(self.lr_pl),
                                                variable_list=self.gen_vars,
                                                n_accum=exp_config.n_accum_grads)

        self.cri_opt = grad_accum_optimizer_gan(loss=self.cri_loss,
                                                optimizer=self._get_optimizer(self.lr_pl),
                                                variable_list=self.cri_vars,
                                                n_accum=exp_config.n_accum_grads)

        # The clip op is only used if not using improved training
        self.d_clip_op = self.clip_op(clip_min=exp_config.clip_min, clip_max=exp_config.clip_max)

        self.global_step = tf.train.get_or_create_global_step()
        self.increase_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))


        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.saver_best = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)



    def critic_loss(self):

        """
        Get the critic loss as defined in the Wasserstein GAN paper, with optional gradient penalty as described
        in the "improved training of Wasserstein GAN paper". 
        """

        cri_loss = tf.reduce_mean(self.D) - tf.reduce_mean(self.D_)

        # if using improved training
        if self.exp_config.improved_training:

            critic_net = self.exp_config.critic_net

            if self.mode3D:
                epsilon_shape = [tf.shape(self.x_c1)[0], 1, 1, 1, 1]
            else:
                epsilon_shape = [tf.shape(self.x_c1)[0], 1, 1, 1]

            epsilon = tf.random_uniform(epsilon_shape, 0.0, 1.0)

            x_hat = epsilon * self.x_c0 + (1 - epsilon) * self.y_c0_
            d_hat = critic_net(x_hat, self.training_pl_cri, scope_reuse=True)

            grads = tf.gradients(d_hat, x_hat)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
            penalty = self.exp_config.scale * tf.reduce_mean(tf.square(slopes - 1.0))

            cri_loss += penalty

        return cri_loss


    def generator_loss(self):

        """
        Get generator loss as described in the Wasserstein GAN paper. Also defines the L1 reg penalty term which
        will be recorded no matter what the value of l1_map_reg to tensorboard.
        """

        gen_loss = tf.reduce_mean(self.D_)

        if not self.exp_config.use_tanh:
            self.l1_map_reg = tf.reduce_mean(tf.abs(self.M))  # Set the term no matter what for TB summaries
        else:
            self.l1_map_reg = tf.reduce_mean(tf.abs(self.y_c0_ - self.x_c1))

        if self.exp_config.l1_map_weight is not None:
            gen_loss += self.l1_map_reg*self.exp_config.l1_map_weight

        return gen_loss


    def clip_op(self, clip_min=-0.01, clip_max=0.01, verbose=False):
        """
        TF operation to clip the weights of the critic as in the original WGAN paper
        (This is only used when improved_training=False)
        :param clip_min: min clip value
        :param clip_max: max clip value
        :param verbose: Display variable summary (for debugging)
        :return: TF operation
        """

        train_variables = tf.trainable_variables()
        d_clip_op = [v.assign(tf.clip_by_value(v, clip_min, clip_max)) for v in self.cri_vars]

        if verbose:
            print('== CLIPPED VARIABLES SUMMARY ==')
            [print(v.name) for v in train_variables if v.name.startswith("critic") and '_bn' not in v.name]

        return d_clip_op


    def train(self):

        """
        Main function for training the VA-GAN model
        :return: 
        """

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

        self.best_wasserstein_distance = -np.inf
        for step in range(self.init_step, self.exp_config.max_iterations):

            # If learning rate is scheduled
            if schedule_lr and step > 0 and step % self.exp_config.divide_lr_frequency == 0:
                self.curr_lr /= 10.0
                logging.info('Updating learning rate to: %f' % self.curr_lr)

            # Set how many critic steps there should be for this generator step
            c_iters = self.exp_config.critic_iter
            if step % self.exp_config.critic_retune_frequency == 0 \
                    or step < self.exp_config.critic_initial_train_duration:
                c_iters = self.exp_config.critic_iter_long

            # Train critic
            logging.info('Doing **critic** update steps (%d steps)' % c_iters)
            for _ in range(c_iters):

                batch_dims = [self.batch_size] + list(self.exp_config.image_size) + [1]
                feed_dict = {self.x_c1: np.zeros(batch_dims),  # dummy variables will be replaced in optimizer
                             self.x_c0: np.zeros(batch_dims),
                             self.training_pl_cri: True,
                             self.training_pl_gen: True,
                             self.lr_pl: self.curr_lr}

                self.cri_opt.do_training_step(sess=self.sess,
                                              sampler_c0=self.sampler_c0,
                                              sampler_c1=self.sampler_c1,
                                              batch_size=self.exp_config.batch_size,
                                              feed_dict=feed_dict,
                                              c0_pl=self.x_c0,
                                              c1_pl=self.x_c1)

                if not self.exp_config.improved_training:
                    self.sess.run(self.d_clip_op)

            # Train generator
            logging.info('Doing **generator** update step')

            batch_dims = [self.batch_size] + list(self.exp_config.image_size) + [1]
            feed_dict = {self.x_c1: np.zeros(batch_dims),  # dummy variables will be replaced in optimizer
                         self.x_c0: np.zeros(batch_dims),
                         self.training_pl_cri: True,
                         self.training_pl_gen: True,
                         self.lr_pl: self.curr_lr}

            self.gen_opt.do_training_step(sess=self.sess,
                                          sampler_c0=self.sampler_c0,
                                          sampler_c1=self.sampler_c1,
                                          batch_size=self.exp_config.batch_size,
                                          feed_dict=feed_dict,
                                          c0_pl=self.x_c0,
                                          c1_pl=self.x_c1)

            # Do tensorboard updates, model validations and other house keeping
            if step % self.exp_config.update_tensorboard_frequency == 0:
                self._update_tensorboard(step)

            if step > 0 and step % self.exp_config.validation_frequency == 0:
                self._do_validation_and_save_model(step)

            if step % self.exp_config.save_frequency == 0:
                # Save latest model
                self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'), global_step=step)

            self.sess.run(self.increase_global_step)


    def predict_mask(self, input_image):

        """
        Get the estimated mask for an input_image
        """

        pred_mask = self.sess.run(self.M, feed_dict={self.x_c1: input_image, self.training_pl_gen: False})
        return pred_mask


    def load_weights(self, log_dir=None, type='latest', **kwargs):

        """
        Load weights into the model
        :param log_dir: experiment directory into which all the checkpoints have been written
        :param type: can be 'latest', 'best_wasserstein' (highest validation Wasserstein distance), or 'iter' (specific
                     iteration, requires passing the iteration argument with a valid step number from the checkpoint 
                     files)
        """

        if not log_dir:
            log_dir = self.log_dir

        if type=='latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type=='best_wasserstein':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_wasserstein.ckpt')
        elif type=='iter':
            assert 'iteration' in kwargs, "argument 'iteration' must be provided for type='iter'"
            iteration = kwargs['iteration']
            init_checkpoint_path = os.path.join(log_dir, 'model.ckpt-%d' % iteration)
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_wasserstein/iter.' % type)

        self.saver.restore(self.sess, init_checkpoint_path)



    ### HELPER FUNCTIONS ############################################################################################

    def _setup_log_dir_and_continue_mode(self):

        """
        Set up logging directories and if an experiment with the same name already exists continue training that 
         model under the name <original_name>_cont 
        """

        # Default values
        self.log_dir = os.path.join(sys_config.log_root, 'gan', self.exp_config.experiment_name)
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


    def _get_optimizer(self, lr_pl):

        """
        Helper function for getting the right optimizer
        """

        if self.exp_config.optimizer_handle == tf.train.AdamOptimizer:
            return self.exp_config.optimizer_handle(lr_pl, beta1=self.exp_config.beta1, beta2=self.exp_config.beta2)
        else:
            return self.exp_config.optimizer_handle(lr_pl)


    def _make_tensorboard_summaries(self):

        """
        Create all the tensorboard summaries for the scalars and images. 
        """

        # tensorboard summaries
        tf.summary.scalar('learning_rate', self.lr_pl)

        tf.summary.scalar('l1_reg_term', self.l1_map_reg)

        if self.exp_config.rescale_to_one:
            display_range = [-1, 1]
        else:
            display_range = [None, None]

        def _image_summaries(prefix, y_c0_, x_c1, x_c0):

            if len(self.img_tensor_shape) == 5:
                data_dimension = 3
            elif len(self.img_tensor_shape) == 4:
                data_dimension = 2
            else:
                raise ValueError('Invalid image dimensions')

            if data_dimension == 3:
                y_c0_disp = y_c0_[:, :, :, self.exp_config.image_z_slice, :]
                x_c1_disp = x_c1[:, :, :, self.exp_config.image_z_slice, :]
                x_c0_disp = x_c0[:, :, :, self.exp_config.image_z_slice, :]
            else:
                y_c0_disp = y_c0_
                x_c1_disp = x_c1
                x_c0_disp = x_c0

            sum_gen = tf.summary.image('%s_a_generated_CN' % prefix, tf_utils.put_kernels_on_grid(
                y_c0_disp,
                min_int=display_range[0],
                max_int=display_range[1],
                batch_size=self.exp_config.batch_size))
            sum_c1 = tf.summary.image('%s_example_AD' % prefix, tf_utils.put_kernels_on_grid(
                x_c1_disp,
                min_int=display_range[0],
                max_int=display_range[1],
                batch_size=self.exp_config.batch_size))
            sum_c0 = tf.summary.image('%s_example_CN' % prefix, tf_utils.put_kernels_on_grid(
                x_c0_disp,
                min_int=display_range[0],
                max_int=display_range[1],
                batch_size=self.exp_config.batch_size))

            difference_map_pl = tf.abs(y_c0_disp - x_c1_disp)
            sum_dif = tf.summary.image('%s_b_difference_CN' % prefix,
                                       tf_utils.put_kernels_on_grid(difference_map_pl,
                                                                    batch_size=self.exp_config.batch_size))

            return tf.summary.merge([sum_gen, sum_c1, sum_c0, sum_dif])

        _image_summaries('train', self.y_c0_, self.x_c1, self.x_c0)

        tf.summary.scalar('critic_loss', self.cri_loss)
        tf.summary.scalar('generator_loss', self.gen_loss)

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary_op = tf.summary.merge_all()

        # validation summaries
        self.val_disc_loss_pl = tf.placeholder(tf.float32, shape=[], name='disc_val_loss')
        disc_val_summary_op = tf.summary.scalar('validation_critic_loss', self.val_disc_loss_pl)

        self.val_gen_loss_pl = tf.placeholder(tf.float32, shape=[], name='gen_val_loss')
        gen_val_summary_op = tf.summary.scalar('validation_generator_loss', self.val_gen_loss_pl)

        # val images
        img_val_summary_op = _image_summaries('val', self.y_c0_, self.x_c1, self.x_c0)

        self.val_summary_op = tf.summary.merge([disc_val_summary_op, gen_val_summary_op, img_val_summary_op])



    def _update_tensorboard(self, step):

        """
        Update the tensorboard summaries at a given step. Also print some logging information. 
        """

        c1_imgs = self.sampler_c1(self.exp_config.batch_size)
        c0_imgs = self.sampler_c0(self.exp_config.batch_size)

        g_loss_train, d_loss_train, summary_str = self.sess.run([self.gen_loss, self.cri_loss, self.summary_op],
                                                                feed_dict={self.x_c1: c1_imgs,
                                                                           self.x_c0: c0_imgs,
                                                                           self.training_pl_cri: True,
                                                                           self.training_pl_gen: True,
                                                                           self.lr_pl: self.curr_lr}
        )

        self.summary_writer.add_summary(summary_str, step)
        self.summary_writer.flush()

        logging.info( "[Step: %d], generator loss: %g, critic_loss: %g" % (step, g_loss_train, d_loss_train))


    def _do_validation_and_save_model(self, step):

        """
        Evaluate model on the validation set and save the required checkpoints at a given step. 
        """

        total_g_loss_val = 0
        total_d_loss_val = 0

        for _ in range(self.exp_config.num_val_batches):
            c1_imgs = self.data.validationAD.next_batch(self.exp_config.batch_size)[0]
            c0_imgs = self.data.validationCN.next_batch(self.exp_config.batch_size)[0]

            g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.cri_loss],
                                                   feed_dict={self.x_c1: c1_imgs,
                                                              self.x_c0: c0_imgs,
                                                              self.training_pl_cri: False,
                                                              self.training_pl_gen: False})

            total_d_loss_val += d_loss_val
            total_g_loss_val += g_loss_val

        total_d_loss_val /= self.exp_config.num_val_batches
        total_g_loss_val /= self.exp_config.num_val_batches

        c1_imgs = self.data.validationAD.next_batch(self.exp_config.batch_size)[0]
        c0_imgs = self.data.validationCN.next_batch(self.exp_config.batch_size)[0]
        validation_summary_str = self.sess.run(self.val_summary_op,
                                               feed_dict={self.val_disc_loss_pl: total_d_loss_val,
                                                          self.val_gen_loss_pl: total_g_loss_val,
                                                          self.x_c0: c0_imgs,
                                                          self.x_c1: c1_imgs,
                                                          self.training_pl_gen: False,
                                                          self.training_pl_cri: False})

        self.summary_writer.add_summary(validation_summary_str, step)
        self.summary_writer.flush()

        logging.info("[Validation], generator loss: %g, critic_loss: %g" % (total_g_loss_val, total_d_loss_val))

        if step > 50 and total_d_loss_val >= self.best_wasserstein_distance:  # in the first steps the W distance is meaningless
            self.best_wasserstein_distance = total_d_loss_val

            logging.info("Found new best Wasserstein distance! Saving model...")

            best_file = os.path.join(self.log_dir, 'model_best_wasserstein.ckpt')
            self.saver_best.save(self.sess, best_file, global_step=step)