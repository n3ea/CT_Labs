import copy
import itertools

import numpy as np

def REF(matr):
    rows, cols = matr.shape
    lead = 0                                            # ведущий элемент

    for row in range(rows):                             # идем по строкам
        if lead >= cols:                                # условие выхода
            return matr

        i = row                                         # за i обозначаем текущую строчку
        while matr[i, lead] == 0:                       # поиск строки с ненулевым элементом в текущем столбце
            i += 1                                      # двигаемся по строкам
            if i == rows:
                i = row
                lead += 1                               # строка заканчивается -> переходим на след столбец
                if lead == cols:
                    return matr

        matr[[row, i]] = matr[[i, row]]                 # обмен строк row и i (нашли ведущий, закидываем строку с ним выше)
        for i in range(row + 1, rows):                  # сложение по модулю 2 с нижними строками, где под ведущим элементом тоже единицы
            if matr[i, lead] != 0:
                matr[i] = (matr[i] + matr[row]) % 2

        lead += 1

    return matr

def RREF(matr):
    rows, cols = matr.shape
    lead = 0                                            # ведущий элемент

    for row in range(rows):                             # идем по строкам
        if lead >= cols:                                # условие выхода
            return matr

        i = row                                         # за i обозначаем текущую строчку
        while matr[i, lead] == 0:                       # поиск строки с ненулевым элементом в текущем столбце
            i += 1                                      # двигаемся по строкам
            if i == rows:
                i = row
                lead += 1                               # строка заканчивается -> переходим на след столбец
                if lead == cols:
                    return matr


        matr[[row, i]] = matr[[i, row]]                 # обмен строк row и i (нашли ведущий, закидываем строку с ним выше)
        for i in range(rows):                           # сложение по модулю 2 со всеми строками, где в столбце с ведущим элементом тоже единицы
            if i != row and matr[i, lead] == 1:
                matr[i] = (matr[i] + matr[row]) % 2

        lead += 1

    return matr

def isZero(matr, i):
    return np.all(matr[i] == 0)

def delZeros(matr):
    non_zero = [not isZero(matr, i) for i in range(matr.shape[0])]
    return matr[non_zero]


class LinearCode:
    @property
    def S(self):
        return delZeros(self.__S)

    def __init__(self):
        self.__S = []

    @S.setter
    def S(self, matr):
        self.__S = matr

    def S_REF(self):                            # приводим матрицу к ступечатому виду + чистим от нулевых строк
        return delZeros(REF(self.S))

    def S_RREF(self):
        return delZeros(RREF(self.S))

    def n(self):                                # столбцы
        return len(self.S[0])

    def k(self):                                # строки
        return len(self.S)

    def lead(self):                             # фиксация ведущих столбцов
        lead = []
        for row in self.S_RREF():
            for i, value in enumerate(row):
                if value != 0:
                    if i not in lead:
                        lead.append(i)
                    break  # Переходим к следующей строке после нахождения ведущего элемента
        return lead

    def X(self):                               # сокращенная матрица X (удалили ведущие столбцы)
        X = self.S_RREF()
        lead = self.lead()
        return np.delete(X, lead, axis=1)

    def H(self):
        X = self.X()
        lead = self.lead()
        I = np.eye(X.shape[1])
        H = np.zeros((X.shape[0] + X.shape[1], X.shape[1]), dtype=int)
        ix = 0
        ii = 0
        for i in range(H.shape[0]):
            if i in lead:
                H[i, :] = X[ix, :]
                ix += 1
            else:
                H[i, :] = I[ii, :]
                ii += 1
        return H

    def U(self):
        U = []
        G = self.S_REF()
        for j in range(0, len(G[0])):
            u = []
            for i in range(0, len(G)):
                u.append(G[i][j])
            if not located(u, U):
                U.append(u)
        flag = True
        while flag:
            flag = False
            for i in range(0, len(U)):
                for j in range(i + 1, len(U)):
                    if not located(sumV(U[i], U[j]), U):
                        U.append(sumV(U[i], U[j]))
                        flag = True
        return np.array(U)

    def U2(self):
        U = []
        u = []
        G = self.S_REF()
        H = self.H()
        for m in self.U()[0]:
            u.append(0)
        for i in range(0, 2 ** 5):
            v = vM(u, G)
            vH = vM(v, H)
            a = 0
            for k in range(0, len(vH)):
                if vH[k] == 0:
                    a += 1
            if a == len(vH):
                if not located(copy.copy(u), U):
                    U.append(copy.copy(u))
            for j in range(0, len(u)):
                if u[len(u) - j - 1] == 0:
                    u[len(u) - j - 1] = 1
                    break
                else:
                    u[len(u) - j - 1] = 0
        return np.array(U)

    def V(self):
        U = self.U()
        V = []
        G = self.S_REF()
        for i in range(0, len(U)):
            V.append(vM(U[i], G))
        return np.array(V)

    def d(self):
        V = self.V()
        d = len(V[0])
        for i in range(0, len(V) - 1):
            for k in range(i + 1, len(V)):
                a = 0
                for j in range(0, len(V[0])):
                    if V[i][j] != V[k][j]:
                        a += 1
                if d > a:
                    d = a
        return d

    def t(self):
        return self.d() - 1

    def e2(self, indexV):
        v = self.V()[indexV]
        H = self.H()
        for i in range(0, len(v) - 1):
            for j in range(i + 1, len(v)):
                e2 = []
                for k in v:
                    e2.append(0)
                e2[i] = 1
                e2[j] = 1
                ve2 = sumV(v, e2)
                ve2H = vM(ve2, H)
                a = 0
                for k in range(0, len(ve2H)):
                    if ve2H[k] == 0:
                        a += 1
                if a == len(ve2H):
                    return e2

def sumV(v1, v2):
    v3 = []
    for i in range(0, len(v1)):
        v3.append((v1[i] + v2[i]) % 2)
    return np.array(v3)

def located(v, M):
    for i in range(0, len(M)):
        a = 0
        for j in range(0, len(M[0])):
            if v[j] == M[i][j]:
                a += 1
        if a == len(M[0]):
            return True
    return False

def vM(v, M):
    vM = []
    for i in range(0, len(M[0])):
        c = 0
        for j in range(0, len(M)):
            c += M[j][i] * v[j]
        vM.append(c % 2)
    return np.array(vM)

S0 = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
               [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
               [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

LC = LinearCode()
LC.S = S0
print("S matrix:")
print(LC.S)
print("G matrix: ",)
print(LC.S_REF())
print("n =", LC.n())
print("k =", LC.k())
print("G* matrix: ")
print(LC.S_RREF())
print("lead =", LC.lead())
print("X matrix: ")
print(LC.X())
print("H matrix: ")
print(LC.H())
print()
print("U matrix: ")
print(LC.U())
print("U2 matrix: ")
print(LC.U2())
v=LC.V()[0]
print("v = ", v)
vH = vM(v, LC.H())
print("vH = ", vH)
print("d =", LC.d())
print("t =", LC.t())
e1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
print("e1 =", e1)
ve1 = sumV(e1, v)
print("v + e1 =", ve1)
print("(v + e1)@H =", vM(ve1, LC.H()))
e2 = LC.e2(0)
print("e2 =", e2)
ve2 = sumV(e2, v)
print("v + e2 =", ve2)
print("(v + e2)@H =", vM(ve2, LC.H()))