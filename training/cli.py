import click
import os
from typing import List, Tuple
import json
from training.audio import AudioProcessor
from training.model import BugClassifier

def load_dataset(audio_dir: str) -> Tuple[List[str], List[str]]:
    """Load dataset from directory structure.
    
    Expected structure:
    audio_dir/
        bug_type1/
            recording1.wav
            recording2.wav
        bug_type2/
            recording3.wav
            ...
    """
    audio_files = []
    labels = []
    
    for bug_type in os.listdir(audio_dir):
        bug_dir = os.path.join(audio_dir, bug_type)
        if not os.path.isdir(bug_dir):
            continue
            
        for audio_file in os.listdir(bug_dir):
            if audio_file.endswith(('.wav', '.mp3', '.m4a')):
                audio_path = os.path.join(bug_dir, audio_file)
                audio_files.append(audio_path)
                labels.append(bug_type)
    
    return audio_files, labels

@click.group()
def cli():
    """BugID: Identify bugs from audio recordings using machine learning."""
    pass

@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('--model-output', '-m', default='animal_sound_model.tflite',
              help='Path to save trained model')
@click.option('--report-output', '-r', default='training_report.json',
              help='Path to save training report')
def train(dataset_dir: str, model_output: str, report_output: str):
    """Train a new bug classifier model.
    
    DATASET_DIR should contain subdirectories, each named after a bug type,
    containing audio recordings (.wav, .mp3, or .m4a) of that bug type.
    """
    click.echo("Loading dataset...")
    audio_files, labels = load_dataset(dataset_dir)
    
    if not audio_files:
        click.echo("Error: No audio files found in dataset directory", err=True)
        return
    
    click.echo(f"Found {len(audio_files)} audio files across {len(set(labels))} bug types")
    
    # Process audio files
    click.echo("Extracting audio features...")
    processor = AudioProcessor()
    features = processor.process_files(audio_files)
    
    # Train model
    click.echo("Training classifier...")
    classifier = BugClassifier()
    report = classifier.train(features, labels)
    
    # Save model and report
    classifier.save_model(model_output)
    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    click.echo(f"\nModel saved to: {model_output}")
    click.echo(f"Training report saved to: {report_output}")
    
    # Display summary metrics
    click.echo("\nTraining Results:")
    if 'weighted avg' in report:
        avg = report['weighted avg']
        click.echo(f"Accuracy: {report['accuracy']:.3f}")
        click.echo(f"Weighted Precision: {avg['precision']:.3f}")
        click.echo(f"Weighted Recall: {avg['recall']:.3f}")
        click.echo(f"Weighted F1-score: {avg['f1-score']:.3f}")

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--model', '-m', type=click.Path(exists=True), required=True,
              help='Path to trained model file')
def predict(audio_file: str, model: str):
    """Predict bug type from an audio recording."""
    # Load and process audio
    processor = AudioProcessor()
    features = processor.process_file(audio_file)
    print(features)
    # Load model and predict
    classifier = BugClassifier(model_path=model)
    prediction = classifier.predict(features.reshape(1, -1))
    probabilities = classifier.predict_proba(features.reshape(1, -1))
    max_probability = max(probabilities[0])

    print(f"Most likely animal: {prediction[0]}")
    click.echo(f"Confidence: {max_probability:.2%}")
    print(f"Predicted animal probabilities: {probabilities}")

if __name__ == '__main__':
    cli()
