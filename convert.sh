
echo "Converting the following m4a to wav":
for i in *.m4a 
do 
echo "$i";
done

for i in *.m4a 
do 
    echo "Converting $i...";
    out_name="$(basename "$i" .m4a)".wav
    convert_cmd="/c/Program\ Files/FFMpeg/bin/ffmpeg.exe -i \"$i\" \"$out_name\""
    #echo "$convert_cmd"
    eval "$convert_cmd"

done

