# f = open("testcases/yolor-agx/part.csv", "w")
# for i in range(235):
#     f.write(f"{i},0\n")

f_in = open("device_assignment.txt", "r")
f_out = open("device_assignment.csv", "w")
for line in f_in.readlines():
    layername, deviceid = line.split()
    f_out.write(f"{layername},{deviceid}\n")