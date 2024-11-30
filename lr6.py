import numpy as np
import random

# Кодирует сообщение с использованием порождающего полинома
def encode_message(message, generator_poly):
    return np.polymul(message, generator_poly) % 2


# Декодирует сообщение и исправляет ошибки на основе синдрома
def decode_message(encoded_msg, generator_poly, max_error_count, is_block_error):
    syndrome = np.polydiv(encoded_msg, generator_poly)[1] % 2
    return correct_errors_in_message(encoded_msg, syndrome, generator_poly, max_error_count, is_block_error)


# Применяет случайные изменения в биты по указанным позициям
def flip_bits(bits, positions):
    for pos in positions:
        bits[pos] ^= 1  # Изменяем бит на противоположный
    return bits


# Добавляет случайные ошибки в битовую последовательность на случайных позициях
def introduce_random_errors(bits, error_count):
    positions = random.sample(range(len(bits)), error_count)
    print(f"Ошибки добавлены в позиции: {positions}")
    return flip_bits(bits, positions)


# Добавляет блок ошибок в случайный диапазон битов
def introduce_error_block(bits, block_length):
    size = len(bits)
    start = random.randint(0, size - block_length)
    end = (start + block_length - 1) % size
    print(f"Блок ошибок добавлен в диапазоне: {start}-{end}")
    for offset in range(block_length):
        bits[(start + offset) % size] ^= 1  # Меняем биты в этом диапазоне
    return bits


# Удаляет ведущие и завершающие нули из массива
def remove_zeros(arr):
    return np.trim_zeros(np.trim_zeros(arr, 'f'), 'b')


# Проверка, можно ли обнаружить ошибку на основе синдрома
def can_detect_error(syndrome, max_error_count):
    trimmed_syndrome = remove_zeros(syndrome)
    return 0 < len(trimmed_syndrome) <= max_error_count


# Исправляет ошибки в закодированном сообщении, используя синдром и сдвиг
def correct_errors_in_message(encoded_msg, syndrome, generator_poly, max_error_count, is_block_error):
    msg_length = len(encoded_msg)

    # Попытка исправить ошибку с учетом сдвига
    for shift in range(msg_length):
        error_poly = np.zeros(msg_length, dtype=int)
        error_poly[msg_length - shift - 1] = 1
        shifted_syndrome = np.polymul(syndrome, error_poly) % 2
        reduced_syndrome = np.polydiv(shifted_syndrome, generator_poly)[1] % 2

        # Проверка, обнаружима ли ошибка
        if is_block_error and can_detect_error(reduced_syndrome, max_error_count):
            return apply_correction(encoded_msg, reduced_syndrome, shift, generator_poly)
        elif not is_block_error and sum(reduced_syndrome) <= max_error_count:
            return apply_correction(encoded_msg, reduced_syndrome, shift, generator_poly)

    return None  # Если ошибка не исправлена, возвращаем None


# Применяет коррекцию ошибок к закодированному сообщению
def apply_correction(encoded_msg, reduced_syndrome, shift, generator_poly):
    msg_length = len(encoded_msg)
    correction_poly = np.zeros(msg_length, dtype=int)
    correction_poly[shift - 1] = 1
    correction = np.polymul(correction_poly, reduced_syndrome) % 2
    corrected_msg = np.polyadd(correction, encoded_msg) % 2
    return np.array(np.polydiv(corrected_msg, generator_poly)[0] % 2).astype(int)


# Выполняет полный цикл анализа кодирования, добавления ошибок и декодирования
def execute_error_correction_analysis(generator_poly, message, error_func, error_param, max_error_count,
                                      is_block_error):
    print(f"Исходное сообщение: {message}")
    encoded_msg = encode_message(message, generator_poly)
    print(f"Закодированное сообщение: {encoded_msg}")

    # Добавление ошибок
    erroneous_msg = error_func(encoded_msg.copy(), error_param)
    print(f"Сообщение с ошибками: {erroneous_msg}")

    # Декодирование с исправлением ошибок
    decoded_msg = decode_message(erroneous_msg, generator_poly, max_error_count, is_block_error)
    print(f"Декодированное сообщение: {decoded_msg}")

    # Проверка корректности декодированного сообщения
    if np.array_equal(message, decoded_msg):
        print("Декодирование успешно. Сообщение совпадает.\n")
    else:
        print("Ошибка в декодировании. Сообщение не совпадает.\n")


# Тестирует код (7,4) на случайных ошибках
def test_7_4_code():
    print("Тестирование кода (7,4)")
    generator_poly = np.array([1, 1, 0, 1])
    message = np.array([1, 0, 1, 0])

    for error_count in range(1, 4):  # Пробуем с 1, 2 и 3 ошибками
        execute_error_correction_analysis(generator_poly, message, introduce_random_errors, error_count, 1, False)


# Тестирует код (15,9) на блоках ошибок
def test_15_9_code():
    print("Тестирование кода (15,9)")
    generator_poly = np.array([1, 0, 0, 1, 1, 1, 1])
    message = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])

    for block_length in range(1, 5):  # Пробуем с блоками ошибок длиной от 1 до 4
        execute_error_correction_analysis(generator_poly, message, introduce_error_block, block_length, 3, True)

if __name__ == '__main__':
    # Тестирование для кода (7,4)
    test_7_4_code()

    # Тестирование для кода (15,9)
    test_15_9_code()
