from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--gpus', type=int, default=1,
                                 help='Number of gpus to use')
        self.parser.add_argument('--train_dataset', type=str,
                                    help='Path to training dataset')
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_aging', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--input_nc', default=4, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int,
                                 help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int,
                                 help='Output size of generator')

        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int,
                                 help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float,
                                 help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str,
                                 help='Which optimizer to use')
        # Scheduler / stability controls
        self.parser.add_argument('--scheduler_type', default='cosine', type=str,
                                 help='LR scheduler: cosine|none')
        self.parser.add_argument('--warmup_steps', default=500, type=int,
                                 help='LR warmup steps before reaching base LR')
        self.parser.add_argument('--min_lr', default=1e-06, type=float,
                                 help='Minimum LR for cosine decay')
        self.parser.add_argument('--grad_clip_norm', default=1.0, type=float,
                                 help='Clip gradient global norm; <=0 disables')
        self.parser.add_argument('--nan_guard', action='store_true',
                                 help='Skip optimizer step on NaN/Inf grads')
        self.parser.add_argument('--train_encoder', action='store_true',
                                 help='Whether to train the encoder model')
        self.parser.add_argument('--train_decoder', action='store_true',
                                 help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--start_from_encoded_w_plus', action='store_true',
                                 help='Whether to learn residual wrt w+ of encoded image using pretrained pSp.')

        self.parser.add_argument('--lpips_lambda', default=0, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float,
                                 help='W-norm loss multiplier factor')
        self.parser.add_argument('--aging_lambda', default=0, type=float,
                                 help='Aging loss multiplier factor')
        self.parser.add_argument('--cycle_lambda', default=0, type=float,
                                 help='Cycle loss multiplier factor')

        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')

        self.parser.add_argument('--lpips_lambda_aging', default=0, type=float,
                                 help='LPIPS loss multiplier factor for aging')
        self.parser.add_argument('--l2_lambda_aging', default=0, type=float,
                                 help='L2 loss multiplier factor for aging')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to pSp model checkpoint')

        # self.parser.add_argument('--max_steps', default=500000, type=int,
        #                          help='Maximum number of training steps')
        self.parser.add_argument('--max_steps', default=15000, type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int,
                                 help='Validation interval')
        self.parser.add_argument('--save_interval', default=2000, type=int,
                                 help='Model checkpoint interval')

        # arguments for aging
        self.parser.add_argument('--target_age', default=None, type=str,
                                 help='Target age for training. Use `uniform_random` for random sampling of target age')
        self.parser.add_argument('--use_weighted_id_loss', action="store_true",
                                 help="Whether to weight id loss based on change in age (more change -> less weight)")
        self.parser.add_argument('--pretrained_psp_path', default=model_paths['pretrained_psp'], type=str,
                                 help="Path to pretrained pSp network.")

        # arguments for personalization
        self.parser.add_argument('--adaptive_w_norm_lambda', default=1, type=float,
                                 help="Weight for adaptive w-norm loss")
        self.parser.add_argument('--nearest_neighbor_id_loss_lambda', default=1, type=float,
                                 help="Weight for nearest neighbor id loss")
        # curriculum for extrapolation
        self.parser.add_argument('--extrapolation_start_step', default=3000, type=int,
                                 help='Training step to allow any extrapolation')
        self.parser.add_argument('--extrapolation_prob_start', default=0.0, type=float,
                                 help='Start probability of extrapolation once enabled')
        self.parser.add_argument('--extrapolation_prob_end', default=0.5, type=float,
                                 help='Final probability of choosing extrapolation')
        # decoder phase loss scaling
        self.parser.add_argument('--w_norm_lambda_decoder_scale', default=0.5, type=float,
                                 help='Scale w-norm lambda during decoder phase')
        self.parser.add_argument('--aging_lambda_decoder_scale', default=0.5, type=float,
                                 help='Scale aging lambda during decoder phase')
        
        # arguments for resuming training
        self.parser.add_argument('--resume_checkpoint', default=None, type=str,
                                 help='Path to checkpoint to resume training from')
        self.parser.add_argument('--additional_steps', default=10000, type=int,
                                 help='Number of additional steps to train after resuming')
        self.parser.add_argument('--continue_in_same_dir', action='store_true',
                                 help='Continue training in the same directory as the checkpoint')
        
        
    def parse(self):
        opts = self.parser.parse_args()
        return opts
