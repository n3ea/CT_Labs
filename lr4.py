import copy
import random
import numpy as np

# Матрица B для вычислений в коде
B_MATRIX = [
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
]


# Функция генерации порождающей матрицы G
def generate_G():
    return np.concatenate((np.eye(12, dtype=np.int16), B_MATRIX), axis=1)


# Функция генерации проверочной матрицы H
def generate_H():
    return np.concatenate((np.eye(12, dtype=np.int16), B_MATRIX), axis=0)


# Функция для поиска ошибки в кодовом слове
def detect_error(coded_word, H_matrix, B_matrix):
    syndrome = coded_word @ H_matrix % 2
    correction_vector = None

    # Проверка синдрома
    if sum(syndrome) <= 3:
        correction_vector = np.hstack((syndrome, np.zeros(len(syndrome), dtype=int)))
    else:
        for i in range(len(B_matrix)):
            temp = (syndrome + B_matrix[i]) % 2
            if sum(temp) <= 2:
                error_indicator = np.zeros(len(syndrome), dtype=int)
                error_indicator[i] = 1
                correction_vector = np.hstack((temp, error_indicator))

    # Второй этап проверки
    if correction_vector is None:
        syndrome_B = syndrome @ B_matrix % 2
        if sum(syndrome_B) <= 3:
            correction_vector = np.hstack((np.zeros(len(syndrome), dtype=int), syndrome_B))
        else:
            for i in range(len(B_matrix)):
                temp = (syndrome_B + B_matrix[i]) % 2
                if sum(temp) <= 2:
                    error_indicator = np.zeros(len(syndrome), dtype=int)
                    error_indicator[i] = 1
                    correction_vector = np.hstack((error_indicator, temp))

    return correction_vector


# Функция для генерации ошибок в кодовом слове
def generate_random_errors(length, num_errors):
    error_vector = np.zeros(length, dtype=int)
    for _ in range(num_errors):
        while True:
            index = round(random.random() * length) - 1
            if error_vector[index] != 1:
                error_vector[index] = 1
                break
    return error_vector


# Функция для построения матрицы Рида-Маллера
def reed_muller(r, m):
    # Базовые случаи рекурсии
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    if r == m:
        base_matrix = reed_muller(m - 1, m)
        bottom_row = np.zeros((1, 2 ** m), dtype=int)
        bottom_row[0, -1] = 1
        return np.vstack([base_matrix, bottom_row])

    # Рекурсивное построение
    upper_part = np.hstack([reed_muller(r, m - 1), reed_muller(r, m - 1)])
    lower_part = np.hstack(
        [np.zeros((reed_muller(r - 1, m - 1).shape[0], reed_muller(r - 1, m - 1).shape[1]), dtype=int),
         reed_muller(r - 1, m - 1)])
    return np.vstack([upper_part, lower_part])


# Функция для произведения Кронекера
def kronecker_product(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    result = np.zeros((rows_A * rows_B, cols_A * cols_B), dtype=A.dtype)

    for i in range(rows_A):
        for j in range(cols_A):
            result[i * rows_B:(i + 1) * rows_B, j * cols_B:(j + 1) * cols_B] = A[i, j] * B
    return result


# Функция для применения матрицы Кронекера к матрице H
def apply_kronecker_H(H_matrix, m, i):
    base_matrix = np.eye(2 ** (m - i), dtype=int)
    base_matrix = kronecker_product(base_matrix, H_matrix)
    base_matrix = kronecker_product(base_matrix, np.eye(2 ** (i - 1)))
    return base_matrix


# Функция для исправления ошибок в коде Рида-Маллера
def correct_reed_muller_error(original_message, received_word, m):
    H_matrix = np.array([[1, 1], [1, -1]])
    word_array = []
    word_array.append(np.dot(received_word, apply_kronecker_H(H_matrix, m, 1)))
    for i in range(2, m + 1):
        word_array.append(np.dot(word_array[-1], apply_kronecker_H(H_matrix, m, i)))

    max_value = word_array[0][0]
    max_index = -1
    for i in range(len(word_array)):
        for j in range(len(word_array[i])):
            if abs(word_array[i][j]) > abs(max_value):
                max_value = word_array[i][j]
                max_index = j

    # Проверка на неоднозначность исправления
    if sum(1 for i in range(len(word_array)) for j in range(len(word_array[i])) if
           abs(word_array[i][j]) == abs(max_value)) > 1:
        print("Невозможно исправить ошибку!\n")
        return

    # Формирование исправленного сообщения
    corrected_message = list(map(int, list(('{' + f'0:0{m}b' + '}').format(max_index))))
    corrected_message.append(1 if max_value > 0 else 0)
    print("Исправленное сообщение:", np.array(corrected_message[::-1]))

    # Проверка корректности исправления
    if not np.array_equal(original_message, corrected_message):
        print("Сообщение было декодировано с ошибкой!\n")


# Пример первой части задачи
def first_task():
    print('Часть 1: Порождающие и проверочные матрицы, коррекция ошибок')

    G = generate_G()
    print("\nПорождающая матрица G = ")
    print(G)

    H = generate_H()
    print("\nПроверочная матрица H = ")
    print(H)

    u = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    print("\nИсходное слово u = ")
    print(u)

    v = np.dot(u, G)
    print("\nКодовое слово v = ")
    print(v)

    errors_1 = generate_random_errors(len(v), 1)
    print("\nОшибка e1 = ")
    print(errors_1)

    word_with_error_1 = np.add(v, errors_1)
    print("\nКодовое слово с одной ошибкой = ")
    print(word_with_error_1)

    syndrome_1 = np.dot(word_with_error_1, H) % 2
    print("\nСиндром кодового слова с одной ошибкой = ")
    print(syndrome_1)

    correction = detect_error(word_with_error_1, H, B_MATRIX)
    if correction is None:
        print("Ошибка обнаружена, исправить невозможно!")
    else:
        corrected_word = (word_with_error_1 + correction) % 2
        print("\nИсправленное кодовое слово = ", corrected_word)
        print("\nПроверка после исправления = ")
        print(np.dot(corrected_word, H) % 2)

    # Повторение для 2, 3 и 4 ошибок
    for num_errors in [2, 3, 4]:
        errors = generate_random_errors(len(v), num_errors)
        print(f"\nОшибка e{num_errors} = ")
        print(errors)
        word_with_error = np.add(v, errors)
        print(f"\nКодовое слово с {num_errors} ошибками = ")
        print(word_with_error)

        syndrome = np.dot(word_with_error, H) % 2
        print(f"\nСиндром кодового слова с {num_errors} ошибками = ")
        print(syndrome)

        correction = detect_error(word_with_error, H, B_MATRIX)
        if correction is None:
            print("Ошибка обнаружена, исправить невозможно!")
        else:
            corrected_word = (word_with_error + correction) % 2
            print(f"\nИсправленное кодовое слово с {num_errors} ошибками = ", corrected_word)
            print("\nПроверка после исправления = ")
            print(np.dot(corrected_word, H) % 2)


# Пример второй части задачи
def second_task():
    print('Часть 2: Код Рида-Маллера')

    # Код Рида-Маллера с параметрами r=1, m=3
    r, m = 1, 3
    print(f"Код Рида-Маллера: (r = {r}, m = {m})")

    G = reed_muller(r, m)
    print("\nПорождающая матрица G = ")
    print(G)

    u = np.array([1, 1, 0, 0])
    print("Исходное слово u = ")
    print(u)
    encoded_word = np.dot(u, G) % 2
    print("Закодированное слово: ", encoded_word)

    errors = generate_random_errors(2 ** m, 1)
    print("Однократная ошибка: ", errors)
    word_with_error = (encoded_word + errors) % 2
    print("Слово с однократной ошибкой: ", word_with_error)
    correct_reed_muller_error(u, word_with_error, m)

    errors = generate_random_errors(2 ** m, 2)
    print("Двухкратная ошибка: ", errors)
    word_with_error = (encoded_word + errors) % 2
    print("Слово с двухкратной ошибкой: ", word_with_error)
    correct_reed_muller_error(u, word_with_error, m)

    # Повторение для более высоких значений m
    for m_val in [4]:
        r, m = 1, m_val
        print(f"\nКод Рида-Маллера: (r = {r}, m = {m})")
        G = reed_muller(r, m)
        print("\nПорождающая матрица G = ")
        print(G)

        u = np.array([1, 1, 0, 0, 1])
        print("Исходное слово u = ")
        print(u)
        encoded_word = np.dot(u, G) % 2
        print("Закодированное слово: ", encoded_word)

        for num_errors in [1, 2, 3, 4]:
            errors = generate_random_errors(2 ** m, num_errors)
            print(f"\nОшибка e{num_errors} = ", errors)
            word_with_error = (encoded_word + errors) % 2
            print(f"Слово с {num_errors} ошибками: ", word_with_error)
            correct_reed_muller_error(u, word_with_error, m)


# Запуск первой и второй частей задачи
first_task()
second_task()