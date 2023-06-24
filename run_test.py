import evaluation as ev

#all the semeval tests are performed here

ok = 0
notok = 0

with open("notok.txt", "w") as file:
    for x in range(50):
        if ev.sense_check(x):
            ok += 1
        else:
            notok += 1
            file.write(str(x) + "\n")


precision = ok/(ok + notok)
#recall is always 1 here since the model is just doing mcq
f1 = (2*precision*1)/(precision+1)

#print all the stats
print("ok: " + str(ok))
print("notok: " + str(notok))
print("precision: " + str(precision))
print("f1 socre: " + str(f1))
