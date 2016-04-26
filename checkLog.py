import matplotlib.pyplot as plt
def generateLossCurve(fileName):
    f = open(fileName)
    defText = f.read().split('\n')
    defText = defText[1:-1]
    x = []
    y = []    
    for line in defText:
        x += [int(line.split(')')[0])]
        y += [float(line.split(')')[1].split('=')[1])]
    f.close()
    plt.plot(x, y, 'k')
    plt.show()

generateLossCurve('runlog.txt')
