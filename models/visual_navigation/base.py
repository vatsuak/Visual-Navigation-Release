import tensorflow as tf

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn


class VisualNavigationModelBase(BaseModel):
    """
    A model used for navigation that receives, among other inputs,
    an image as its observation of the environment.
    """
    
    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.image_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)

    def _optimal_labels(self, raw_data):
        """
        Return the optimal label based on raw_data.
        """
        raise NotImplementedError

    def _goal_position(self, raw_data):
        """
        Return the goal position (x, y) in egocentric
        coordinates.
        """
        return raw_data['goal_position_ego_n2']

    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        
        if self.p.data_processing.input_processing_function is not None:
            raw_data = self.preprocess_nn_input(raw_data, is_training)

        # Get the input image (n, m, k, d)
        # batch size n x (m x k pixels) x d channels
        img_nmkd = raw_data['img_nmkd']

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        goal_position = self._goal_position(raw_data)
        state_features_n4 = tf.concat(
            [goal_position, raw_data['vehicle_controls_nk2'][:, 0]], axis=1)

        # Optimal Supervision
        optimal_labels_n = self._optimal_labels(raw_data)

        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [img_nmkd, state_features_n4]
        data['labels'] = optimal_labels_n
        return data
    
    def make_processing_functions(self):
        """
        Initialize the processing functions if required.
        """
        # Initialize the distortion function
        if self.p.data_processing.input_processing_function in ['distort_images', 'normalize_distort_images']:
            from training_utils.data_processing.distort_images import basic_image_distortor
            self.image_distortor = basic_image_distortor(self.p.data_processing.input_processing_params.p)

    def preprocess_nn_input(self, raw_data, is_training):
        """
        Pre-process the NN input.
        """
        if is_training:
            # Distort images if required
            if self.p.data_processing.input_processing_function in ['distort_images', 'normalize_distort_images']:
                raw_data['img_nmkd'] = self.image_distortor.augment_images(raw_data['img_nmkd'])
        
        # Normalize images if required
        if self.p.data_processing.input_processing_function in ['normalize_images', 'normalize_distort_images']:
            from training_utils.data_processing.normalize_images import rgb_normalize
            raw_data = rgb_normalize(raw_data)
        
        return raw_data
