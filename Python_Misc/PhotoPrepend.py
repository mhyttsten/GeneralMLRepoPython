import os

os.chdir("/Users/magnushyttsten/Desktop/Lottas_Pictures_New_Deal/Todo_Dated")


def change_dates():
    hour = 0
    minute = 1
    is_first = True
    for f in os.listdir("."):
        if not f.startswith("0") \
                and not f.startswith("1") \
                and not f.startswith("2") \
                and not f.startswith("3") \
                and not f.startswith("4") \
                and not f.startswith("5") \
                and not f.startswith("6") \
                and not f.startswith("7") \
                and not f.startswith("8") \
                and not f.startswith("9"):
            print("*** Will not process file: {}".format(f))
            continue

        d = "1990:01:01 %02d:%02d:00" % (hour,minute)
        s = "exiftool -F -AllDates=\"%s\" %s" % (d,f)
        print s
        if minute == 59:
            minute = 0
            hour += 1
        else:
            minute += 1
        os.system(s)
change_dates()


def rm_year_and_adjust():
    for f in os.listdir("."):
        new_filename = ""
        if f.startswith("19"):
            continue
            # nf = f[2:]
            # year = int(nf[:2])
            # year_adj = year - 40
            # print("Year: {}, adjusted: {}".format(year, year_adj))
            # new_filename = "%d%s" % (year_adj, f[4:])
            # print("Old: %s, new: %s" % (f, new_filename))
        elif f.startswith("200"):
            new_filename = ("6%s" % f[3:])
            print("Old: %s, new: %s" % (f, new_filename))
        # os.rename(f, new_filename)

#--------
def add_year():
    for f in os.listdir("."):
        new_filename = ""
        if f[0] == "8":
            new_filename = "19%s" % f
        elif f[0] == "9":
            new_filename = "19%s" % f
        elif f[0] == "0":
            new_filename = "20%s" % f
        else:
            print("*** Wont process: {}".format(f))
            continue
        print("{} -> {}".format(f, new_filename))
        # os.rename(f, new_filename)



