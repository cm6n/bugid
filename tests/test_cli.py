import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from click.testing import CliRunner
from training.cli import cli, load_dataset
from training.feature_selection import plot_features, plot_mfcc_2d, plot_features_comparison

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        
    def test_load_dataset(self):
        """Test dataset loading from directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset structure
            os.makedirs(os.path.join(tmpdir, 'cricket'))
            os.makedirs(os.path.join(tmpdir, 'beetle'))
            
            # Create dummy audio files with different supported formats
            open(os.path.join(tmpdir, 'cricket', 'sound1.wav'), 'w').close()
            open(os.path.join(tmpdir, 'cricket', 'sound2.mp3'), 'w').close()
            open(os.path.join(tmpdir, 'beetle', 'sound1.m4a'), 'w').close()
            
            # Test loading
            files, labels = load_dataset(tmpdir)
            
            self.assertEqual(len(files), 3)
            self.assertEqual(len(labels), 3)
            self.assertEqual(labels.count('cricket'), 2)
            self.assertEqual(labels.count('beetle'), 1)
    
    @patch('training.cli.AudioProcessor')
    @patch('training.cli.BugClassifier')
    def test_train_command(self, mock_classifier, mock_processor):
        """Test train command workflow."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_files.return_value = MagicMock()
        
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        mock_classifier_instance.train.return_value = {
            'accuracy': 0.95,
            'weighted avg': {
                'precision': 0.95,
                'recall': 0.95,
                'f1-score': 0.95
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset
            os.makedirs(os.path.join(tmpdir, 'cricket'))
            open(os.path.join(tmpdir, 'cricket', 'sound1.wav'), 'w').close()
            
            # Run command
            result = self.runner.invoke(cli, ['train', tmpdir])
            
            self.assertEqual(result.exit_code, 0)
            mock_processor_instance.process_files.assert_called_once()
            mock_classifier_instance.train.assert_called_once()
            
    @patch('training.cli.AudioProcessor')
    @patch('training.cli.BugClassifier')
    def test_predict_command(self, mock_classifier, mock_processor):
        """Test predict command workflow."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_file.return_value = MagicMock()
        
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        mock_classifier_instance.predict.return_value = ['cricket']
        mock_classifier_instance.predict_proba.return_value = [[0.9, 0.1]]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock audio file and model
            audio_path = os.path.join(tmpdir, 'test.wav')
            model_path = os.path.join(tmpdir, 'model.joblib')
            open(audio_path, 'w').close()
            open(model_path, 'w').close()
            
            # Run command
            result = self.runner.invoke(cli, ['predict', audio_path, '--model', model_path])
            
            self.assertEqual(result.exit_code, 0)
            mock_processor_instance.process_file.assert_called_once()
            mock_classifier_instance.predict.assert_called_once()
            mock_classifier_instance.predict_proba.assert_called_once()
            
    @patch('training.cli.plot_features')
    @patch('training.cli.AudioProcessor')
    def test_features_command(self, mock_processor, mock_plot_features):
        """Test features command workflow."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock audio file
            audio_path = os.path.join(tmpdir, 'test.wav')
            open(audio_path, 'w').close()
            
            # Run command
            result = self.runner.invoke(cli, ['features', audio_path])
            
            self.assertEqual(result.exit_code, 0)
            mock_plot_features.assert_called_once()
    
    @patch('training.cli.plot_mfcc_2d')
    @patch('training.cli.AudioProcessor')
    def test_mfcc_command(self, mock_processor, mock_plot_mfcc_2d):
        """Test mfcc command workflow."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock audio file
            audio_path = os.path.join(tmpdir, 'test.wav')
            open(audio_path, 'w').close()
            
            # Run command
            result = self.runner.invoke(cli, ['mfcc', audio_path])
            
            self.assertEqual(result.exit_code, 0)
            mock_plot_mfcc_2d.assert_called_once()
    
    @patch('training.cli.plot_features_comparison')
    @patch('training.cli.AudioProcessor')
    def test_compare_command(self, mock_processor, mock_plot_features_comparison):
        """Test compare command workflow."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock audio files
            audio_path1 = os.path.join(tmpdir, 'test1.wav')
            audio_path2 = os.path.join(tmpdir, 'test2.wav')
            open(audio_path1, 'w').close()
            open(audio_path2, 'w').close()
            
            # Run command with multiple files
            result = self.runner.invoke(cli, ['compare', audio_path1, audio_path2, 
                                             '--labels', 'Bug A', 'Bug B'])
            
            self.assertEqual(result.exit_code, 0)
            mock_plot_features_comparison.assert_called_once()
            
            # Check that the arguments were passed correctly
            args, kwargs = mock_plot_features_comparison.call_args
            self.assertEqual(len(args[0]), 2)  # Two audio files
            self.assertEqual(len(args[1]), 2)  # Two labels

if __name__ == '__main__':
    unittest.main()
