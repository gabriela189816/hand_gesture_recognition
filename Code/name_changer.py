import os
# PATH
path ="/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture"
list_dir = os.listdir(path)
print(len(list_dir))

# OLD NAME
i = 0
for file in list_dir:
    current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/"+str(file)
    new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/palm_"+str(i)+".png"
    os.rename(current_file_name, new_file_name)
    i += 1

print("Done")