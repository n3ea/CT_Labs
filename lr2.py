
import random


X = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
]

def Ik(k):
    # Создаем единичную матрицу размера k x k
    return [[1 if j == i else 0 for j in range(k)] for i in range(k)]

def sum_vector(V):
    # Возвращаем сумму элементов вектора V
    return sum(V)

def vM(v, M):
    # Умножаем вектор v на матрицу M
    return [sum(M[j][i] * v[j] for j in range(len(M))) % 2 for i in range(len(M[0]))]

def sumV(v1, v2):
    # Суммируем два вектора поэлементно по модулю 2
    return [(v1[i] + v2[i]) % 2 for i in range(len(v1))]

def located(v, M):
    # Проверяем, находится ли вектор v в матрице M
    return any(v == M[i] for i in range(len(M)))

def locatedIndex(v, M):
    # Возвращаем индекс вектора v в матрице M
    for i in range(len(M)):
        if v == M[i]:
            return i
    return -1

def attachHorizontal(M1, M2):
    # Соединяем две матрицы M1 и M2 по горизонтали
    return [M1[i] + M2[i] for i in range(len(M1))]

def attachVertical(M1, M2):
    # Соединяем две матрицы M1 и M2 по вертикали
    return M1 + M2

def genX(k, n):
    # Генерируем матрицу X размера k x n с определенными условиями
    while True:
        # Генерируем случайные строки
        X = [[random.randint(0, 1) for _ in range(n)] for _ in range(k)]

        # Проверяем, чтобы в каждой строке было не менее 4 единиц
        if all(sum(row) >= 4 for row in X) and \
                all(sumV(X[i], X[j]).count(1) >= 3 for i in range(k) for j in range(i + 1, k)) and \
                all(sumV(sumV(X[i], X[j]), X[m]).count(1) >= 2 for i in range(k) for j in range(i + 1, k) for m in
                    range(j + 1, k)) and \
                all(sumV(sumV(sumV(X[i], X[j]), X[m]), X[l]).count(1) >= 1 for i in range(k) for j in range(i + 1, k)
                    for m in range(j + 1, k) for l in range(m + 1, k)):
            return X

def U(g):
    # Генерируем базис U из генератора g
    U = []
    for j in range(len(g[0])):
        u = [g[i][j] for i in range(len(g))]  # Собираем столбец в вектор u
        if not located(u, U):
            U.append(u)
    flag = True
    while flag:  # Пока есть новые векторы для добавления
        flag = False
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                new_vector = sumV(U[i], U[j])
                if not located(new_vector, U):  # Если сумма нового вектора не найдена в U
                    U.append(new_vector)
                    flag = True
    return U

def e(n, errors):
    # Генерируем вектор ошибок длиной n с заданным количеством ошибок
    e = [0] * n
    indices = random.sample(range(n), errors)  # Выбираем случайные индексы для ошибок
    for idx in indices:
        e[idx] = 1  # Устанавливаем ошибку в выбранные индексы
    return e

def correctWord(H, sindrom, word):
    # Исправляем слово, используя синдром и матрицу проверки четности H для 1 ошибки
    i = locatedIndex(sindrom, H)  # Находим индекс синдрома в H
    if i != -1:
        word[i] ^= 1  # Исправляем слово, меняя соответствующий бит
    else:
        print("Такого синдрома нет в матрице Н")
    return word

def correctWord2mistakes(H, sindrom, word):
    # Исправляем слово, используя синдром и матрицу проверки четности H для 2 ошибок
    k, d = -1, -1
    for i in range(len(H)):
        if locatedIndex(sindrom, H) == i:  # Находим первый синдром
            k = i
            break
        for j in range(i + 1, len(H)):
            if located(sindrom, [sumV(H[i], H[j])]):  # Проверяем комбинации синдромов
                k, d = i, j
    if k != -1:  # Если нашли синдром
        word[k] ^= 1
        if d != -1:
            word[d] ^= 1
    else:
        print("Такого синдрома нет в матрице синдромов")
    return word
def print_matrix(matrix, name):
    # Печатаем матрицу с заданным именем
    print(f"\n{name} = ")
    for row in matrix:
        print(row)

def V(u, g):
    # Генерируем кодовое слово V из базиса U и матрицы генераторов G
    return [vM(u[i], g) for i in range(len(u))]

def first():
    K, N = 4, 7
    print("\n1Часть----------\n")

    print_matrix(X, "X")

    G = attachHorizontal(Ik(K), X)  # Создаем генераторную матрицу G
    print_matrix(G, "G")

    H = attachVertical(X, Ik(N - K))  # Создаем матрицу проверки четности H
    print_matrix(H, "H")

    U1 = U(G)  # Генерируем базис U из G
    print_matrix(U1, "U")

    V1 = V(U1, G)  # Получаем кодовые слова V из U1 и G
    v = V1[0]  # Берем первое кодовое слово
    print_matrix([v], "кодовое слово")

    e1 = e(N, 1)  # Генерируем вектор ошибок с одной ошибкой
    print_matrix([e1], "e1")

    word1 = sumV(v, e1)  # Создаем слово с одной ошибкой
    print_matrix([word1], "кодовое слово с одной ошибкой")

    sindrom1 = vM(word1, H)
    print_matrix([sindrom1], "синдром кодового слова с одной ошибкой")

    print_matrix([correctWord(H, sindrom1, word1)], "исправленное кодовое слово c одной ошибкой")  # Исправляем слово и печатаем
    print_matrix([vM(word1, H)], "проверка")

    e2 = e(N, 2)  # Генерируем вектор ошибок с двумя ошибками
    print_matrix([e2], "e2")

    word2 = sumV(v, e2)  # Создаем слово с двумя ошибками
    print_matrix([word2], "кодовое слово с двумя ошибками")

    sindrom2 = vM(word2, H)
    print_matrix([sindrom2], "синдром кодового слова с двумя ошибками")

    print_matrix([correctWord2mistakes(H, sindrom2, word2)], "исправленное кодовое слово c двумя ошибками")
    print_matrix([vM(word2, H)], "проверка")

def second():
    N, K = 11, 4
    print("\n2Часть----------\n")

    Xsecond = genX(K, N - K)  # Генерируем новую матрицу X
    print_matrix(Xsecond, "X")

    G = attachHorizontal(Ik(K), Xsecond)  # Создаем генераторную матрицу G
    print_matrix(G, "G")

    H = attachVertical(Xsecond, Ik(N - K))  # Создаем матрицу проверки четности H
    print_matrix(H, "H")

    U2 = U(G)  # Генерируем базис U из G
    print_matrix(U2, "U")

    V2 = V(U2, G)  # Получаем кодовые слова V из U2 и G
    v = V2[0]
    print_matrix([v], "кодовое слово")

    for errors in [1, 2, 3]:  # Проходим по количеству ошибок
        e_vec = e(N, errors)  # Генерируем вектор ошибок
        print_matrix([e_vec], f"e{errors}")

        word = sumV(v, e_vec)  # Создаем слово с ошибками
        print_matrix([word], f"кодовое слово с {errors} ошибками")

        sindrom = vM(word, H)  # Получаем синдром слова с ошибками
        print_matrix([sindrom], f"синдром кодового слова с {errors} ошибками")

        print_matrix([correctWord2mistakes(H, sindrom, word)], f"исправленное кодовое слово c {errors} ошибками")
        print_matrix([vM(word, H)], "проверка")

first()
second()
