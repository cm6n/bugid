cd training
python cli.py train ../../animal_sounds --model-output=../../animal_sound_model.tflite

cp ../../animal_sound_model.tflite ../../../../AndroidStudioProjects/bugid/app/src/main/assets/animal_sound_model.tflite

cp ../../animal_sound_model_classes.txt ../../../../AndroidStudioProjects/bugid/app/src/main/assets/animal_names.txt

cd -
