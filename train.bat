cd training
python cli.py train ../../bugsounds --model-output=../../animal_sound_model.tflite && cp ../../animal_sound_model.tflite ../../../../AndroidStudioProjects/bugid/app/src/main/assets/animal_sound_model.tflite
cd --