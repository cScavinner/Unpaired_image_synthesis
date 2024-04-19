import options.base_options as base_options

class TestOptions(base_options.BaseOptions):
    '''
    This class encompasses the test options and those defined in BaseOptions.
    '''

    def initialize(self, parser):
        parser = base_options.BaseOptions.initialize(self, parser)
        parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
        parser.add_argument('-m', '--model', help='Path to the model folder', type=str, required=True)
        parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
        parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (0,0,0))
        parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
        parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
        parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='equinus')
        parser.add_argument('--gpu', help='GPU to use (If -1, goes on CPU)', type=int,required=False, default=0)

        subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

        mDRIT_parser = subparsers.add_parser("mDRIT", help="mDRIT++ mode parser")
        mDRIT_parser.add_argument('--mode', help='Mode to use: reconstruction or degradation', type=str,required=False, default='reconstruction')
        mDRIT_parser.add_argument('--latents', help='Get latent variables', action='store_true')
        
        self.isTrain= False
        return parser
