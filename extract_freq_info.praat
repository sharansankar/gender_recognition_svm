

Read from file: "test_audio.wav"

 
 
To Pitch (cc)... 0.0 75.0 15 'yes' 0.03 0.45 0.01 0.35 0.14 600.0

duration = Get end time

min = Get minimum: 0.0, duration, "Hertz", "Parabolic"
max = Get maximum: 0.0, duration, "Hertz", "Parabolic"
mean = Get mean: 0.0, duration, "Hertz"

writeFileLine: "output.txt", mean,", ",min, ", ", max

Quit