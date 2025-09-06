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
        self.parser.add_argument('--coach', default=None, type=str,
                                 help='Coach to use: orig|orig_nn|tune_no_psp|tune_psp|delta')
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
        self.parser.add_argument('--seed', default=None, type=int,
                                 help='Global random seed for reproducibility (optional)')
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
        self.parser.add_argument('--val_start_step', default=0, type=int,
                                 help='Skip validation until this global step')
        self.parser.add_argument('--save_interval', default=2000, type=int,
                                 help='Model checkpoint interval')

        # validation control
        self.parser.add_argument('--disable_validation', action='store_true',
                                 help='Disable validation and test logging during training')
        self.parser.add_argument('--val_disable_aging', action='store_true',
                                 help='Validation uses reconstruction (no aging) for stability')
        self.parser.add_argument('--val_max_batches', default=0, type=int,
                                 help='If >0, limit validation to this many batches per run')
        self.parser.add_argument('--val_deterministic', action='store_true',
                                 help='Use fixed RNG seeds during validation for reproducibility')

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
        # Contrastive impostor (age-aware) toggles
        self.parser.add_argument('--contrastive_id_lambda', type=float, default=0.0,
                                 help='Weight for impostor-only age-aware contrastive ID loss; 0 disables.')
        self.parser.add_argument('--mb_index_path', type=str, default="",
                                 help='Path to bank .pt (e.g., banks/ffhq_ir50_age_5y.pt). Empty disables.')
        self.parser.add_argument('--mb_k', type=int, default=64,
                                 help='Negatives per sample.')
        self.parser.add_argument('--mb_apply_min_age', type=int, default=None,
                                 help='Min target age to apply the contrastive loss (inclusive). None = no lower bound.')
        self.parser.add_argument('--mb_apply_max_age', type=int, default=None,
                                 help='Max target age to apply the contrastive loss (inclusive). None = no upper bound.')
        self.parser.add_argument('--mb_bin_neighbor_radius', type=int, default=0,
                                 help='How many neighboring 5y bins to include on each side (0 = same bin only).')
        self.parser.add_argument('--mb_temperature', type=float, default=0.07,
                                 help='Temperature for cosine similarities.')
        # FAISS miner toggles
        self.parser.add_argument('--mb_use_faiss', action='store_true',
                                 help='Use FAISS miner for negatives.')
        self.parser.add_argument('--mb_top_m', type=int, default=512,
                                 help='Top-M candidates per query before filtering.')
        self.parser.add_argument('--mb_min_sim', type=float, default=0.20,
                                 help='Lower cosine bound for semi-hard negatives.')
        self.parser.add_argument('--mb_max_sim', type=float, default=0.70,
                                 help='Upper cosine bound for negatives.')
        # Miner preset profile (optional)
        self.parser.add_argument('--mb_profile', type=str, default='custom',
                                 choices=['baseline', 'soft', 'soft32', 'custom'],
                                 help='Preset for miner band and K. custom = use individual flags.')
        # ROI-ID micro-loss toggles
        self.parser.add_argument('--roi_id_lambda', type=float, default=0.0,
                                 help='Weight for ROI-ID (eyes+mouth) identity loss; 0 disables.')
        self.parser.add_argument('--roi_size', type=int, default=112,
                                 help='Crop size (pixels) for IR-SE50.')
        self.parser.add_argument('--roi_pad', type=float, default=0.35,
                                 help='Padding ratio around tight eye/mouth boxes (e.g., 0.35 = +35%).')
        self.parser.add_argument('--roi_jitter', type=float, default=0.08,
                                 help='Uniform jitter fraction for box center/size during train (0.08 = ±8%).')
        self.parser.add_argument('--roi_landmarks_model', type=str, default="",
                                 help='Optional path to Dlib 68-landmark model. Empty = try autodetect or heuristic fallback.')
        self.parser.add_argument('--roi_use_mouth', action='store_true',
                                 help='Include mouth ROI in ROI-ID (default off if flag absent).')
        self.parser.add_argument('--roi_use_eyes', action='store_true',
                                 help='Include eyes ROI in ROI-ID (default off if flag absent).')
        # ROI-ID schedule controls
        self.parser.add_argument('--roi_id_schedule_s1', type=str, default=None,
                                 help='Stage-1 schedule for ROI-ID lambda as "step:value,..." (e.g., "0:0.05,20000:0.07,36000:0.05").')
        self.parser.add_argument('--roi_id_lambda_s2', type=float, default=None,
                                 help='Stage-2 fixed ROI-ID lambda; falls back to --roi_id_lambda if unset.')
        # Geometry loss (shape ratios from 68 landmarks)
        self.parser.add_argument('--geom_lambda', type=float, default=0.0,
                                 help='Weight for geometry ratio loss; 0 disables (default).')
        self.parser.add_argument('--geom_stage', type=str, default='s1', choices=['s1', 's2', 'both'],
                                 help='Apply geometry loss in Stage-1, Stage-2, or both (default: s1).')
        self.parser.add_argument('--geom_parts', type=str, default='eyes,nose,mouth',
                                 help='Comma-separated subset of parts to include: eyes,nose,mouth')
        self.parser.add_argument('--geom_weights', type=str, default='1.0,0.6,0.4',
                                 help='Comma-separated weights for parts (eyes,nose,mouth order).')
        self.parser.add_argument('--geom_norm', type=str, default='interocular', choices=['interocular'],
                                 help="Normalization for ratios; only 'interocular' currently supported.")
        self.parser.add_argument('--geom_huber_delta', type=float, default=0.03,
                                 help='Huber threshold (in ratio units) for geometry loss.')
        self.parser.add_argument('--geom_landmarks_model', type=str, default='',
                                 help='Optional override path to Dlib 68-landmarks model for geometry/ROI.')
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

        # EMA (Exponential Moving Average) controls
        self.parser.add_argument('--ema', action='store_true',
                                 help='Enable EMA tracking of selected module weights during training (off by default).')
        self.parser.add_argument('--ema_decay', type=float, default=0.999,
                                 help='EMA decay factor. Effective only if --ema is set.')
        self.parser.add_argument('--ema_scope', type=str, default='decoder', choices=['decoder', 'decoder+adapter', 'all'],
                                 help='Which modules to track with EMA. "decoder" (default), "decoder+adapter", or "all" (trainable modules in this stage).')
        self.parser.add_argument('--eval_with_ema', dest='eval_with_ema', action='store_true',
                                 help='Use EMA weights during validation/eval when EMA is enabled.')
        self.parser.add_argument('--no_eval_with_ema', dest='eval_with_ema', action='store_false',
                                 help='Disable using EMA weights during validation/eval.')
        self.parser.set_defaults(eval_with_ema=True)
        
        
    def parse(self):
        opts = self.parser.parse_args()
        # Apply miner presets if requested; profile wins over explicit flags
        profile = getattr(opts, 'mb_profile', 'custom')
        if profile and profile != 'custom':
            if profile == 'baseline':
                opts.mb_k = 64
                opts.mb_min_sim = 0.20
                opts.mb_max_sim = 0.70
            elif profile == 'soft':
                opts.mb_k = 48
                opts.mb_min_sim = 0.25
                opts.mb_max_sim = 0.65
            elif profile == 'soft32':
                opts.mb_k = 32
                opts.mb_min_sim = 0.25
                opts.mb_max_sim = 0.65
            else:
                # Unknown profile (should not happen due to choices); ignore
                pass
        return opts
