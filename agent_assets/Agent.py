import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import agent_assets.A_hparameters as hp
from datetime import datetime
from os import path, makedirs
import numpy as np
from agent_assets.replaybuffer import ReplayBuffer
from tensorflow.keras import mixed_precision
from functools import partial
import tensorflow_probability as tfp

#leave memory space for opencl
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

keras.backend.clear_session()

class Player():
    """A agent class which plays the game and learn.

    Main Algorithm
    --------------
    V-MPO
    """
    def __init__(self, observation_space, action_space, model_f, m_dir=None,
                 log_name=None, start_step=0, mixed_float=False):
        """
        Parameters
        ----------
        observation_space : gym.Space
            Observation space of the environment.
        action_space : gym.Space
            Action space of the environment. Current agent expects only
            a discrete action space.
        model_f
            A function that returns actor, critic models. 
            It should take obeservation space and action space as inputs.
            It should not compile the model.
        m_dir : str
            A model directory to load the model if there's a model to load
        log_name : str
            A name for log. If not specified, will be set to current time.
            - If m_dir is specified yet no log_name is given, it will continue
            counting.
            - If m_dir and log_name are both specified, it will load model from
            m_dir, but will record as it is the first training.
        start_step : int
            Total step starts from start_step
        mixed_float : bool
            Whether or not to use mixed precision
        """
        # model : The actual training model
        # t_model : Fixed target model
        print('Model directory : {}'.format(m_dir))
        print('Log name : {}'.format(log_name))
        print('Starting from step {}'.format(start_step))
        print(f'Use mixed float? {mixed_float}')
        self.action_space = action_space
        self.action_range = action_space.high - action_space.low
        self.action_shape = action_space.shape
        self.observation_space = observation_space
        self.mixed_float = mixed_float
        if mixed_float:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
        
        assert hp.Algorithm in hp.available_algorithms, "Wrong Algorithm!"

        # Special variables
        if hp.Algorithm == 'V-MPO':
            
            self.eta = tf.Variable(1.0, trainable=True, name='eta',dtype='float32')
            self.alpha_mu = tf.Variable(1.0, trainable=True, name='alpha_mu',
                                        dtype='float32')
            self.alpha_sig = tf.Variable(1.0, trainable=True, name='alpha_sig',
                                        dtype='float32')
        
        elif hp.Algorithm == 'A2C':
            action_num = tf.reduce_prod(self.action_shape)
            self.log_sigma = tf.Variable(tf.fill((action_num),0.1),
                            trainable=True,name='sigma',dtype='float32')

        #Inputs
        if hp.ICM_ENABLE:
            actor, critic, icm_models = model_f(observation_space, action_space)
            encoder, inverse, forward = icm_models
            self.models={
                'actor' : actor,
                'critic' : critic,
                'encoder' : encoder,
                'inverse' : inverse,
                'forward' : forward,
            }
        else:
            actor, critic = model_f(observation_space, action_space)
            self.models={
                'actor' : actor,
                'critic' : critic,
            }
        targets = ['actor', 'critic']

        # Common ADAM optimizer; in V-MPO loss is merged together
        common_lr = tf.function(partial(self._lr, 'common'))
        self.common_optimizer = keras.optimizers.Adam(
            learning_rate=common_lr,
            epsilon=hp.lr['common'].epsilon,
            global_clipnorm=hp.lr['common'].grad_clip,
        )
        if self.mixed_float:
            self.common_optimizer = mixed_precision.LossScaleOptimizer(
                self.common_optimizer
            )

        for name, model in self.models.items():
            lr = tf.function(partial(self._lr, name))
            optimizer = keras.optimizers.Adam(
                learning_rate=lr,
                epsilon=hp.lr[name].epsilon,
                global_clipnorm=hp.lr[name].grad_clip,
            )
            if self.mixed_float:
                optimizer = mixed_precision.LossScaleOptimizer(
                    optimizer
                )
            model.compile(optimizer=optimizer)
            model.summary()
        
        # Load model if specified
        if m_dir is not None:
            for name, model in self.models.items():
                model.load_weights(path.join(m_dir,name))
            print(f'model loaded : {m_dir}')

        # Initialize target model
        self.t_models = {}
        for name in targets:
            model = self.models[name]
            self.t_models[name] = keras.models.clone_model(model)
            self.t_models[name].set_weights(model.get_weights())

        # File writer for tensorboard
        if log_name is None :
            self.log_name = datetime.now().strftime('%m_%d_%H_%M_%S')
        else:
            self.log_name = log_name
        self.file_writer = tf.summary.create_file_writer(path.join('logs',
                                                         self.log_name))
        self.file_writer.set_as_default()
        print('Writing logs at logs/'+ self.log_name)

        # Scalars
        self.start_training = False
        self.total_steps = tf.Variable(start_step, dtype=tf.int64)
        
        # Savefile folder directory
        if m_dir is None :
            self.save_dir = path.join('savefiles',
                            self.log_name)
            self.save_count = 0
        else:
            if log_name is None :
                self.save_dir, self.save_count = path.split(m_dir)
                self.save_count = int(self.save_count)
            else:
                self.save_dir = path.join('savefiles',
                                        self.log_name)
                self.save_count = 0
        self.model_dir = None

    def _lr(self, name):
        effective_steps = self.total_steps - int(hp.lr[name].halt_steps)
        if tf.greater(effective_steps, int(hp.lr[name].nsteps)):
            return hp.lr[name].end
        elif tf.less(effective_steps, 0):
            return 0.0
        else :
            new_lr = hp.lr[name].start*\
                ((hp.lr[name].end/hp.lr[name].start)**\
                    (tf.cast(effective_steps,tf.float32)/hp.lr[name].nsteps))
            return new_lr


    @tf.function
    def pre_processing(self, observation:dict):
        """
        Preprocess input data
        """
        processed_obs = {}
        for name, obs in observation.items():
            obs_range = self.observation_space[name].high - \
                        self.observation_space[name].low
            obs_middle = (self.observation_space[name].high + 
                          self.observation_space[name].low)/2
                          
            # In case some values are not finite
            # range -> 1
            # middle -> 0
            obs_range = np.where(np.isfinite(obs_range),obs_range,1)
            obs_middle = np.where(np.isfinite(obs_middle),obs_middle,0)
            # If only one observation is given, reshape to [1,...]
            if len(observation[name].shape)==\
                len(self.observation_space[name].shape):
                processed_obs[name] = \
                    2*(tf.cast(obs[tf.newaxis,...],tf.float32)-obs_middle)\
                                                            /obs_range
                                        
            else :
                processed_obs[name] = \
                    2*(tf.cast(obs, tf.float32)-obs_middle)/obs_range
        return processed_obs

    @tf.function
    def choose_action(self, before_state):
        """
        Policy part
        """
        processed_state = self.pre_processing(before_state)
        # Policy Gradient methods use old target model to collect data
        
        if hp.Algorithm == 'V-MPO':
            mu, sigma = self.t_models['actor'](processed_state, training=False)
            action_distrib = tfp.distributions.MultivariateNormalTriL(
                loc=mu, scale_tril=sigma, name='choose_action_dist'
            )
        elif hp.Algorithm == 'PPO':
            mu, sigma = self.t_models['actor'](processed_state, training=False)
            action_distrib = tfp.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma, name='choose_action_dist'
            )
        elif hp.Algorithm == 'A2C':
            mu = self.t_models['actor'](processed_state, training=False)
            sigma = tf.exp(self.log_sigma)
            action_distrib = tfp.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma, name='choose_action_dist'
            )
        
        action = action_distrib.sample()
        action = tf.clip_by_value(action, self.action_space.low, self.action_space.high)
        return action


    def act_batch(self, before_state):
        action = self.choose_action(before_state)
        return action.numpy()
        
    def act(self, before_state):
        """
        Will squeeze axis=0 if Batch_num = 1
        If you don't want to squeeze, use act_batch()
        """
        action = self.choose_action(before_state)
        action_np = action.numpy()
        if action_np.shape[0] == 1:
            return action_np[0]
        else:
            return action_np


    @tf.function
    def train_step(self, o, r, d, a, sn_batch, sp_batch=None):
        """
        All inputs are expected to be preprocessed
        """
        batch_size = tf.shape(a)[0]

        #################################################### ICM START
        if hp.ICM_ENABLE:
            with tf.GradientTape() as icm_tape:
                f_s = self.models['encoder'](o, training=True)
                f_sp = self.models['encoder'](sp_batch, training=True)
                a_pred = self.models['inverse']([f_s, f_sp], training=True)
                inverse_loss = tf.reduce_mean(tf.square(a_pred-a))

                f_sp_pred = self.models['forward']([a, f_s])
                # Leave batch axis
                f_sp_flat = tf.reshape(f_sp,(batch_size, -1))
                f_sp_pred_flat = tf.reshape(f_sp_pred,(batch_size, -1))
                r_intrinsic = tf.losses.mse(f_sp_flat, f_sp_pred_flat)
                forward_loss = tf.reduce_mean(r_intrinsic)

                icm_loss = (1-hp.ICM_loss_forward_weight)*inverse_loss + \
                           hp.ICM_loss_forward_weight * forward_loss

                if self.mixed_float:
                    icm_loss = self.models['inverse']\
                                   .optimizer\
                                   .get_scaled_loss(icm_loss)

            encoder_vars = self.models['encoder'].trainable_weights
            inverse_vars = self.models['inverse'].trainable_weights
            forward_vars = self.models['forward'].trainable_weights

            concat_vars = encoder_vars + inverse_vars + forward_vars

            concat_gradients = icm_tape.gradient(icm_loss, concat_vars)
            if self.mixed_float:
                concat_gradients = self.models['inverse']\
                                    .optimizer\
                                    .get_unscaled_gradients(concat_gradients)
            
            self.models['encoder'].optimizer.apply_gradients(
                zip(concat_gradients[:len(encoder_vars)], encoder_vars)
            )
            self.models['inverse'].optimizer.apply_gradients(
                zip(concat_gradients[len(encoder_vars):
                                     len(encoder_vars)+len(inverse_vars)], 
                                     inverse_vars)
            )
            self.models['forward'].optimizer.apply_gradients(
                zip(concat_gradients[-len(forward_vars):], forward_vars)
            )

            r += hp.ICM_intrinsic * r_intrinsic

            if self.total_steps % hp.log_per_steps==0:
                tf.summary.scalar('Max_r_i',tf.reduce_max(r_intrinsic), 
                                    self.total_steps)
        ###################################################### ICM END


        with tf.GradientTape() as tape:
            ###################################################### IQN START
            if hp.IQN_ENABLE:
                with tape.stop_recording():
                    tau = tf.random.uniform([batch_size, hp.IQN_SUPPORT])
                    tau_inv = 1.0 - tau
                    nth_critic_input = sn_batch.copy()
                    # add tau to input
                    nth_critic_input['tau'] = tau
                    # In On-Policy, on-line critic is used to calculate target
                    nth_support = self.models['critic'](
                        nth_critic_input, 
                        training=False,
                    )
                    # Shape (batch, support)
                    G = r[...,tf.newaxis] + \
                        tf.cast(tf.math.logical_not(d),
                        tf.float32)[...,tf.newaxis]*\
                        (hp.Q_discount**hp.Buf.N) * \
                        nth_support

                critic_input = o.copy()
                critic_input['tau'] = tau
                support = self.models['critic'](
                    critic_input,
                    training=True,
                )
                # For logging
                v = tf.math.reduce_mean(support, axis=-1)
                # Shape (batch, support, support)
                # One more final axis, because huber reduces one final axis
                huber_loss = \
                    keras.losses.huber(G[...,tf.newaxis,tf.newaxis],
                                    support[:,tf.newaxis,:,tf.newaxis])
                mask = (G[...,tf.newaxis] -\
                            support[:,tf.newaxis,:]) >= 0.0
                tau_expand = tau[:,tf.newaxis,:]
                tau_inv_expand = tau_inv[:,tf.newaxis,:]
                raw_loss = tf.where(
                    mask, tau_expand * huber_loss, tau_inv_expand * huber_loss
                )
                # Shape (batch,)
                critic_unweighted_loss = tf.reduce_mean(
                    tf.reduce_sum(raw_loss, axis=-1),
                    axis=-1
                )
            ###################################################### IQN END

            ###################################################### non-IQN START
            else:
                with tape.stop_recording():
                    nth_v = self.models['critic'](
                        sn_batch,
                        training=False
                    )
                    G = r +\
                        tf.cast(tf.math.logical_not(d),tf.float32) *\
                        (hp.Q_discount**hp.Buf.N) *\
                        nth_v
                v = self.models['critic'](
                    o,
                    training=True,
                )
                critic_unweighted_loss = tf.math.square(v-G)
            ###################################################### non-IQN END

            L_V = tf.math.reduce_mean(critic_unweighted_loss)/2

            if hp.IQN_ENABLE:
                G = tf.reduce_mean(G, axis=-1)

            if hp.Algorithm == 'V-MPO':
                with tape.stop_recording():
                    if hp.IQN_ENABLE:
                        iqn_input = o.copy()
                        tau = tf.random.uniform([batch_size, hp.IQN_SUPPORT])
                        iqn_input['tau'] = tau
                        support_target = self.t_models['critic'](
                            iqn_input,
                            training=False
                        )
                        v_target = tf.reduce_mean(support_target,axis=-1)
                    else:
                        v_target = self.t_models['critic'](
                            o,
                            training=False,
                        )
                    # (B,)
                    adv_target = G - v_target

                    mu_t, sig_t = self.t_models['actor'](o, training=False)
                    target_dist = tfp.distributions.MultivariateNormalTriL(
                        loc=mu_t, scale_tril=sig_t, name='target_dist'
                    )

                mu, sig = self.models['actor'](o, training=True)
                online_dist = tfp.distributions.MultivariateNormalTriL(
                    loc=mu, scale_tril=sig, name='online_dist'
                )
                online_logprob = online_dist.log_prob(a)
                
                # Top half advantages
                top_i = tf.argsort(adv_target, 
                            direction='DESCENDING')[:tf.shape(adv_target)[0]//2]
                adv_top_half = tf.cast(tf.gather(adv_target, top_i),'float32')
                online_logprob_top_half = tf.gather(online_logprob, top_i)
                # (B/2,)
                phi = tf.nn.softmax(adv_top_half/tf.stop_gradient(self.eta))
                L_PI = tf.math.reduce_mean(-phi * online_logprob_top_half)

                eta_term = tf.math.log(tf.reduce_mean(tf.math.exp(
                    adv_top_half/self.eta
                )))
                is_finite = tf.cast(tf.math.is_finite(eta_term),'float32')

                L_ETA = self.eta*hp.VMPO_eps_eta + \
                        self.eta*tf.math.multiply_no_nan(eta_term,is_finite)
                
                online_dist_mu = tfp.distributions.MultivariateNormalTriL(
                    loc=mu, scale_tril=sig_t, name='online_dist_mu'
                )
                online_dist_sig = tfp.distributions.MultivariateNormalTriL(
                    loc=mu_t, scale_tril=sig, name='online_dist_sig'
                )

                KL_mu = tfp.distributions.kl_divergence(
                    target_dist, online_dist_mu, allow_nan_stats=True,
                )
                KL_sig = tfp.distributions.kl_divergence(
                    target_dist, online_dist_sig, allow_nan_stats=True,
                )

                KL_mu_safe = tf.math.multiply_no_nan(
                    KL_mu,
                    tf.cast(
                        tf.math.is_finite(KL_mu), 'float32'
                    )
                )
                KL_sig_safe = tf.math.multiply_no_nan(
                    KL_sig,
                    tf.cast(
                        tf.math.is_finite(KL_sig), 'float32'
                    )
                )

                L_A_mu = tf.reduce_mean(
                    self.alpha_mu*(hp.VMPO_eps_alpha_mu\
                                    -tf.stop_gradient(KL_mu_safe))
                    + tf.stop_gradient(self.alpha_mu)*KL_mu_safe
                )
                L_A_sig = tf.reduce_mean(
                    self.alpha_sig*(hp.VMPO_eps_alpha_sig\
                                    -tf.stop_gradient(KL_sig_safe))
                    + tf.stop_gradient(self.alpha_sig)*KL_sig_safe
                )

                loss = L_V + L_PI + L_ETA + L_A_mu + L_A_sig

            elif hp.Algorithm == 'PPO':
                with tape.stop_recording():
                    mu_t, sig_t = self.t_models['actor'](o, training=False)
                    target_dist = tfp.distributions.MultivariateNormalDiag(
                        loc=mu_t, scale_diag=sig_t, name='target_dist'
                    )
                    target_logprob = target_dist.log_prob(a)

                mu, sig = self.models['actor'](o, training=True)
                online_dist = tfp.distributions.MultivariateNormalDiag(
                    loc=mu, scale_diag=sig, name='online_dist'
                )
                online_logprob = online_dist.log_prob(a)
                ratios = tf.exp(online_logprob - target_logprob)

                adv = G - tf.stop_gradient(v)
                surrogate1 = ratios * adv
                surrogate2 = tf.clip_by_value(
                    ratios, 1-hp.PPO_eps_clip, 1+hp.PPO_eps_clip
                ) * adv
                L_PI = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                loss = L_V + L_PI
            
            elif hp.Algorithm == 'A2C':
                mu = self.models['actor'](o, training=True)
                sigma = tf.exp(self.log_sigma)
                online_dist = tfp.distributions.MultivariateNormalDiag(
                    loc=mu, scale_diag=sigma, name='online_dist'
                )
                online_logprob = online_dist.log_prob(a)

                adv = G - v
                
                L_PI = -tf.reduce_mean(online_logprob*tf.stop_gradient(adv))
                loss = L_V + L_PI


            original_loss = loss
            if self.mixed_float:
                loss = self.common_optimizer.get_scaled_loss(loss)

        critic_vars = self.models['critic'].trainable_weights
        actor_vars = self.models['actor'].trainable_weights
        all_vars = critic_vars + actor_vars

        if hp.Algorithm == 'V-MPO':
            vmpo_vars = [self.eta, self.alpha_mu, self.alpha_sig]
            all_vars = all_vars+vmpo_vars
        elif hp.Algorithm == 'A2C':
            all_vars.append(self.log_sigma)

        all_gradients = tape.gradient(loss, all_vars)
        if self.mixed_float:
            all_gradients = \
                self.common_optimizer.get_unscaled_gradients(
                    all_gradients
                )
            # Mixed precision handles NaN grads itself
            apply_grad = True
        else:
            apply_grad = True
            # Do not update if any grad is not finite
            for grad in all_gradients:
                apply_grad = tf.logical_and(
                    apply_grad,
                    tf.math.reduce_all(tf.math.is_finite(grad))
                )
        if apply_grad:
            self.common_optimizer.apply_gradients(
                zip(all_gradients, all_vars)
            )

        if hp.Algorithm == 'V-MPO':
            # Clip eta, alpha
            self.eta.assign(
                tf.reduce_max([self.eta, hp.VMPO_eta_min]))
            self.alpha_mu.assign(
                tf.reduce_max([self.alpha_mu, hp.VMPO_alpha_min]))
            self.alpha_sig.assign(
                tf.reduce_max([self.alpha_sig, hp.VMPO_alpha_min]))
        elif hp.Algorithm == 'A2C':
            self.log_sigma.assign(
                tf.math.log(tf.clip_by_value(tf.exp(self.log_sigma), 
                            hp.A2C_sig_min, hp.A2C_sig_max))
            )


        if self.total_steps % hp.log_per_steps==0:
            tf.summary.scalar('L_V', L_V, self.total_steps)
            tf.summary.scalar('L_pi', L_PI, self.total_steps)
            tf.summary.scalar('Total_loss',original_loss, self.total_steps)
            tf.summary.scalar('MaxV', tf.reduce_max(v), self.total_steps)
            tf.summary.scalar('Log_prob',tf.reduce_mean(online_logprob),
                                            self.total_steps)
            if hp.Algorithm == 'V-MPO':
                tf.summary.scalar('L_eta', L_ETA, self.total_steps)
                tf.summary.scalar('L_alpha_mu', L_A_mu, self.total_steps)
                tf.summary.scalar('L_alpha_sig', L_A_sig, self.total_steps)
                tf.summary.scalar('eta', self.eta, self.total_steps)
                tf.summary.scalar('alpha_mu', self.alpha_mu, self.total_steps)
                tf.summary.scalar('alpha_sig', self.alpha_sig,self.total_steps)
                tf.summary.scalar('KL_mu', tf.reduce_mean(KL_mu), 
                                                self.total_steps)
                tf.summary.scalar('KL_sig', tf.reduce_mean(KL_sig), 
                                                    self.total_steps)
                tf.summary.scalar('adv_top_half',tf.reduce_mean(adv_top_half),
                                                self.total_steps)
            elif hp.Algorithm == 'A2C':
                tf.summary.scalar('MaxSigma', 
                    tf.reduce_max(tf.exp(self.log_sigma)), self.total_steps)


        if self.total_steps % hp.log_grad_per_steps == 0:
            tf.summary.scalar(
                'critic_grad_norm',
                tf.linalg.global_norm(all_gradients[:len(critic_vars)]),
                step=self.total_steps,
            )
            if hp.Algorithm == 'V-MPO':
                tf.summary.scalar(
                    'actor_grad_norm',
                    tf.linalg.global_norm(all_gradients[len(critic_vars):-3]),
                    step=self.total_steps,
                )
                tf.summary.scalar(
                    'eta_grad',
                    all_gradients[-3],
                    step=self.total_steps,
                )
                tf.summary.scalar(
                    'alpha_mu_grad',
                    all_gradients[-2],
                    step=self.total_steps,
                )
                tf.summary.scalar(
                    'alpha_sig_grad',
                    all_gradients[-1],
                    step=self.total_steps,
                )
            elif hp.Algorithm == 'PPO':
                tf.summary.scalar(
                    'actor_grad_norm',
                    tf.linalg.global_norm(all_gradients[len(critic_vars):]),
                    step=self.total_steps,
                )
            elif hp.Algorithm == 'A2C':
                tf.summary.scalar(
                    'actor_grad_norm',
                    tf.linalg.global_norm(all_gradients[len(critic_vars):-1]),
                    step=self.total_steps,
                )



    def step(self, buf:ReplayBuffer):
        if self.total_steps % hp.log_per_steps==0:
            tf.summary.scalar(f'lr_common', self._lr('common'),self.total_steps)
            for name in self.models:
                tf.summary.scalar(f'lr_{name}',self._lr(name),self.total_steps)

        if self.total_steps % hp.histogram == 0:
            for model in self.models.values():
                for var in model.trainable_weights:
                    tf.summary.histogram(var.name, var, step=self.total_steps)

        s_batch, a_batch, r_batch, d_batch, sn_batch, sp_batch \
            = buf.sample(need_next_obs=hp.ICM_ENABLE)
        s_batch = self.pre_processing(s_batch)
        sn_batch = self.pre_processing(sn_batch)
        if hp.ICM_ENABLE:
            sp_batch = self.pre_processing(sp_batch)

        # sp_batch is only used with ICM enabled
        data = (
            s_batch,
            r_batch, 
            d_batch, 
            a_batch, 
            sn_batch,
            sp_batch,
        )

        self.train_step(*data)

        # Hard Target update
        if self.total_steps % hp.Target_update == 0:
            for t_model_name in self.t_models:
                model = self.models[t_model_name]
                t_model = self.t_models[t_model_name]
                t_model.set_weights(model.get_weights())
            target_updated=True
        else:
            target_updated = False

        self.total_steps.assign_add(1)

        return target_updated

    def save_model(self):
        """
        Saves the model and return next save file number
        """
        print('saving model..')
        self.save_count += 1
        self.model_dir = path.join(self.save_dir, str(self.save_count))
        if not path.exists(self.model_dir):
            makedirs(self.model_dir)
        for name, model in self.models.items():
            weight_dir = path.join(self.model_dir,name)
            model.save_weights(weight_dir)

        return self.save_count

