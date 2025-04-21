cd training
python cli.py train ../../bugsounds_small --model-output=../../animal_sound_model_small.tflite

cp ../../animal_sound_model_small.tflite ../../../../AndroidStudioProjects/bugid/app/src/main/assets/animal_sound_model.tflite

cp ../../animal_sound_model_small_classes.txt ../../../../AndroidStudioProjects/bugid/app/src/main/assets/animal_names.txt

cd -
