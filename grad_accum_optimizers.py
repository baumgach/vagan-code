# Optimiser classes that average gradients over multiple mini-batches. Needed to train on large 3D ADNI data
# Author: Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf

class grad_accum_optimizer_base:

    def __init__(self, loss, optimizer, variable_list, n_accum=1, verbose=False):

        self.verbose = verbose

        self.optimizer_handle = optimizer
        self.n_accum = n_accum

        # Initialize variable holding the accumlated gradients and create a zero-initialisation op
        accum_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in variable_list]
        self.zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accum_grad]

        # Calculate gradients and define accumulation op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss, var_list=variable_list)
        self.accum_op = [ac.assign_add(gg[0]) for ac, gg in zip(accum_grad, gradients)]

        # Define the gradient mean op
        self.accum_normaliser_pl = tf.placeholder(dtype=tf.float32, name='accum_normaliser')
        self.mean_op = [ag.assign(tf.divide(ag, self.accum_normaliser_pl)) for ag in accum_grad]

        # Reassemble the gradients in the [value, var] format and do define train op
        final_gradients = [(ag, gg[1]) for ag, gg in zip(accum_grad, gradients)]
        self.train_op = optimizer.apply_gradients(final_gradients)


class grad_accum_optimizer_gan(grad_accum_optimizer_base):

    def do_training_step(self,
                         sess,
                         sampler_c0,
                         sampler_c1,
                         batch_size,
                         feed_dict,
                         c0_pl,
                         c1_pl):

        sess.run(self.zero_op)

        for accum_counter in range(self.n_accum):

            if self.verbose:
                print('Accumulating batch %d/%d' % (accum_counter+1, self.n_accum))

            batch_c0 = sampler_c0(batch_size)
            batch_c1 = sampler_c1(batch_size)

            # update newest imageas in batch
            feed_dict[c0_pl] = batch_c0
            feed_dict[c1_pl] = batch_c1

            sess.run(self.accum_op, feed_dict=feed_dict)

        sess.run(self.mean_op, feed_dict={self.accum_normaliser_pl: self.n_accum})
        sess.run(self.train_op, feed_dict=feed_dict)


class grad_accum_optimizer_classifier(grad_accum_optimizer_base):

    def do_training_step(self,
                         sess,
                         sampler,
                         batch_size,
                         feed_dict,
                         img_pl,
                         lbl_pl,
                         loss=None):

        sess.run(self.zero_op)

        for accum_counter in range(self.n_accum):

            if self.verbose:
                print('Accumulating batch %d/%d' % (accum_counter+1, self.n_accum))

            x, y = sampler(batch_size)

            # update newest images and labels in batch
            feed_dict[img_pl] = x
            feed_dict[lbl_pl] = y

            sess.run(self.accum_op, feed_dict=feed_dict)

        sess.run(self.mean_op, feed_dict={self.accum_normaliser_pl: self.n_accum})

        if loss is not None:
            loss_eval, _ = sess.run([loss, self.train_op], feed_dict=feed_dict)
            return loss_eval

        return sess.run(self.train_op, feed_dict=feed_dict)


