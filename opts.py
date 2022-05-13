import argparse




def get_args():
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 32, help='Evalutatino batch Size')
    optim_args.add_argument('--adam_epsilon', type=float,  default= 1e-8, help='Adam epsilon')
    optim_args.add_argument('--warmup_steps', type=int,  default= 100, help='Warmup Steps')
    optim_args.add_argument('--weight_decay', type=float,  default= 0.04, help='Warmup Steps')
    optim_args.add_argument('--learning_rate', type=float,  default=5e-5, help='Initial Learning rate')
    optim_args.add_argument( "--gradient_accumulation_steps",type=int, default=3, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    optim_args.add_argument( "--val_check_interval",type=float, default=0.5, 
                            help="validation check interval ratio")
    optim_args.add_argument( "--gradient_cliping",type=float, default=0.5, 
                            help=" The value at which to clip gradients ")

    
    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--num_labels', type=int, default=None, help='The number of labels')
    data_args.add_argument('--eval_splits', type=list, default=None, help='eval splits list')
    data_args.add_argument('--task_name', type=str, default= 'kobert', help='Task name')
    data_args.add_argument('--max_seq_length', type=int, default= 128, help='Max sequence legnth')
    data_args.add_argument('--data_dir', type=str, default= '.', help='Directory to load dataset')
    data_args.add_argument('--data_output_dir', type=str, default= '.', help='Directory to store processed dataset')
    data_args.add_argument('--doc_stride', type=int, default= 128, help='Document stride')
    data_args.add_argument('--max_query_length', type=int, default= 64, help='Max query length')
    data_args.add_argument('--max_answer_length', type=int, default= 30, 
                           help='The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another')
    data_args.add_argument('--version_2_with_negative', action="store_true",
                           help="If true, the SQuAD examples contain some that do not have an answer.",)
    data_args.add_argument( '--verbose_logging', action="store_true",
                           help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.",
    )
    data_args.add_argument('--null_score_diff_threshold', type=float,default=0.0, help="If null_score - best_non_null is greater than the threshold predict null.")
    data_args.add_argument('--prediction_file', type=str,default=None, help="Prediction File")
    data_args.add_argument('--data_file', type=str, default= '.', help='Name of dataset file')
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--model_name_or_path', type=str, default="albert-base-v2",
                            help='Model name or path')
    model_args.add_argument('--max_epochs', type=int, default=10, help='Max epoch size')
    model_args.add_argument('--do_lower_case', action="store_true", help="Set this flag if you are using an uncased model."
    )
    model_args.add_argument('--output_dir', type=str, default=".", help='Max epoch size')

    
    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=-1, help='Number of gpu')
    misc_args.add_argument('--checkpoints_dir', type=str, default=None, help='Checkpoint store directory')
    misc_args.add_argument('--checkpoints_dir_callback', type=str, default=None, help='Checkpoint callback directory')
    misc_args.add_argument('--threads', type=int, default=1, help='Number of workers')





    args = parser.parse_args()
    return args
