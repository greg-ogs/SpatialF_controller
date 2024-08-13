import mysql.connector
class sql_queryl:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="greg",
            password="contpass01",
            database="airy"
        )

        self.mycursor = self.mydb.cursor()


    def coord(self):
        # def for lv

        self.mycursor.execute("SELECT * FROM DATA WHERE ID = 1")
        myresult = self.mycursor.fetchall()

        list_one = myresult[0]
        x = list_one[1]
        y = list_one[2]
        signal = list_one[3]
        stop = list_one[4]
        m = [x, y, signal, stop]

        sql = "UPDATE DATA SET X = %s, Y = %s WHERE ID = %s"
        val = (0, 0, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()
        return m

    def signal_0(self):
        # def for lv
        # call before update increments in lv
        # set a timer after lv execute this (4s aprox)
        sql = "UPDATE DATA SET SIGNALS = %s WHERE ID = %s"
        val = (0, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()


def get_coord():
    lqy = sql_queryl()
    m = lqy.coord()
    return m


def get_signal_0():
    lqy = sql_queryl()
    lqy.signal_0()

get_signal_0()