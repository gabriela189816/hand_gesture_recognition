import os
# PATH
person = "P9/"

path ="/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person)
list_dir = os.listdir(path)
print(len(list_dir))

# OLD NAME
i1 = 0
i2 = 0
i3 = 0
i4 = 0
i5 = 0
variable = "p9"

for file in list_dir:
    name = file
    if name[9: 11] == "01":
        new_name1 = variable + "_palm"
        current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person)+str(file)
        new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person)+new_name1+str(i1)+".png"
        os.rename(current_file_name, new_file_name)
        i1 += 1
    elif name[9: 11] == "03":
        new_name2 = variable + "_fist"
        current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person)+str(file)
        new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person)+new_name2+str(i2) + ".png"
        os.rename(current_file_name, new_file_name)
        i2 += 1
    elif name[9: 11] == "06":
        new_name3 = variable + "_index"
        current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + str(file)
        new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + new_name3 + str(i3) + ".png"
        os.rename(current_file_name, new_file_name)
        i3 += 1
    elif name[9: 11] == "07":
        new_name4 = variable + "_ok"
        current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + str(file)
        new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + new_name4 + str(i4) + ".png"
        os.rename(current_file_name, new_file_name)
        i4 += 1
    elif name[9: 11] == "09":
        new_name5 = variable + "_c"
        current_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + str(file)
        new_file_name = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/raw_images/"+str(person) + new_name5 + str(i5) + ".png"
        os.rename(current_file_name, new_file_name)
        i5 += 1
    else:
        print("This is not an image")

print("Done")