from matplotlib import pyplot as plt

y_inf = [0, 15.1098361, 14.9007284, 19.1122749, 19.1122749, 19.1122749]
y_40 = [0, 13.4168196, 17.6416348, 16.9620768, 16.9620768, 16.9620768]
y_16 = [0, 9.35022095, 13.6486017, 14.2592969, 14.2592969, 14.2592969]
y_12 = [0, 12.4574436, 12.4574436, 12.4574436, 12.4574436, 12.4574436]
y_8 = [0, 12.3835421, 8.23262853, 8.23262853, 8.23262853, 8.23262853]
x_ = [1, 2, 3, 4, 5, 6]

plt.plot(x_, y_inf, label="INF")
plt.plot(x_, y_40, label="40Gbps")
plt.plot(x_, y_16, label="16Gbps")
plt.plot(x_, y_12, label="12Gbps")
plt.plot(x_, y_8, label="8Gbps")
leg = plt.legend(loc='lower right')
plt.savefig("speedup.png")