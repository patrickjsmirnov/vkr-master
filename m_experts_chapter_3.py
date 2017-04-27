# для главы 3. 3 метода выбора вектора
# import numpy as np
# import random

import math
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

launch_number = 50
time_finish = 1000

# выигрыши
p1 = [0.3, 0.45, 0.6, 0.4, 0.1]

n = len(p1)

# эксперты
m = 0
p2 = []

# считали подсказки экспертов
def read(f):
	global m
	for line in f:
		row = [float(i) for i in line.split()]
		p2.append(row)
		m += 1

f = open('data.txt', 'r')
read(f)
f.close()

index_best_arm = p1.index(max(p1))

# функция игры
def func_win():
	mean_win[current_number] += bernoulli.rvs(p1[current_number])
	num_of_games[current_number] += 1
	return 0

# инициализация
def func_first_win():
	for i in range(n):
		mean_win[i] = bernoulli.rvs(p1[i])
	return 0

# UCB1
def number_of_arms(t):
	global current_number
	max_temp = mean_win[0] / num_of_games[0] + math.sqrt(2 * math.log(t) / num_of_games[0])
	max_temp_index = 0
	for i in range(n):
		temp[i] = mean_win[i] / num_of_games[i] + math.sqrt(2 * math.log(t) / num_of_games[i])
		if temp[i] > max_temp:
			max_temp = temp[i]
			max_temp_index = i
	current_number = max_temp_index
	num_of_games[current_number] += 1
	return 0

# regret
def func_regret_calculation():
	for i in range(time_finish):
		j, temp_var = 0, 0
		while j < i:
			temp_var += p1[current_number_vector[j]]
			j += 1
		regret_vec[i] = i * p1[index_best_arm] - temp_var
	return 0


#  без эксперта
sum_of_reward = [0 for i in range(n)]
mean_regret_vec = [0 for i in range(time_finish)]
regret_vec = [0 for i in range(time_finish)]
mean_win = [0 for i in range(n)]
num_of_games = [1 for i in range(n)]
current_number_vector = [0 for i in range(time_finish)]
temp = [0 for i in range(n)]


launch_count = 0
while launch_count < launch_number:
	count = 1
	current_number = 0

	for i in range(n):
		mean_win[i] = 0
		num_of_games[i] = 1
		temp[i] = 0

	for i in range(time_finish):
		regret_vec[i] = 0
		current_number_vector[i] = 0

	func_first_win()

	while count < time_finish:
		number_of_arms(count)
		current_number_vector[count] = current_number
		func_win()
		count += 1

	func_regret_calculation()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	for i in range(n):
		sum_of_reward[i] += mean_win[i]

	launch_count += 1

# суммарный выигрыш
sum_reward = 0
for i in range(n):
	sum_reward += sum_of_reward[i] / launch_number


print('sum_reward = ', sum_reward)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


# блок с экспертами
#  функция предсказания
# здесь подсчитаем матрицу предсказаний
def func_forecast():
	for i in range(m):
		for j in range(n):
			current_forecast_matr[i][j] = bernoulli.rvs(p2[i][j])
			forecast_matr[i][j] += current_forecast_matr[i][j]
	return 0

# инициализация подсказок экспертов
def choice_expert_initially(i):
	for j in range(n):
		forecast_vec[j] = forecast_matr[i][j]
	return 0

new_forecast_matr = [[0 for i in range(n)] for j in range(m)]
new_mean = [0 for i in range(n)]

# текущий вектор по мере
def calc_vec_mera(t):
	ro = [0 for i in range(m)]
	for i in range(m):
		kvad_razn = 0
		for j in range(n):
			kvad_razn += (forecast_matr[i][j] / t - mean_win[j] / num_of_games[j]) * (forecast_matr[i][j] / t - mean_win[j] / num_of_games[j])
			new_forecast_matr[i][j] = forecast_matr[i][j] / t
		ro[i] = math.sqrt(kvad_razn)

	for i in range(n):
		new_mean[i] = mean_win[i] / num_of_games[i]
	temp_min = ro[0]
	temp_index_min = 0
	for i in range(m):
		if ro[i] < temp_min:
			temp_min = ro[i]
			temp_index_min = i

	for j in range(n):
		forecast_vec[j] = forecast_matr[temp_index_min][j]
	num_of_games_expert[temp_index_min] += 1
	return 0

# текущий вектор с наилучшими точностями
def calc_vec_best_accr(t):
	accur_matr = [[0 for i in range(n)] for j in range(m)]
	for i in range(m):
		for j in range(n):
			accur_matr[i][j] = abs(forecast_matr[i][j] / t - mean_win[j] / num_of_games[j])

	for i in range(n):
		temp_min = accur_matr[0][i]
		temp_index_min = 0
		for j in range(m):
			if accur_matr[j][i] < temp_min:
				temp_min = accur_matr[j][i]
				temp_index_min = j
		forecast_vec[i] = forecast_matr[temp_index_min][i]
		num_of_choice_accur[temp_index_min][i] += 1
	return 0

# выбор эксперта на основе UCB1
def choice_expert(t):
	global current_number_expert
	max_temp_expert = mean_win_expert[0] / num_of_games_expert[0] + math.sqrt(2 * math.log(t) / num_of_games_expert[0])
	max_temp_index_expert = 0
	for k in range(m):
		temp_expert[k] = mean_win_expert[k] / num_of_games_expert[k] + math.sqrt(2 * math.log(t) / num_of_games_expert[k])
		if temp_expert[k] > max_temp_expert:
			max_temp_expert = temp_expert[k]
			max_temp_index_expert = k

	current_number_expert = max_temp_index_expert
	num_of_games_expert[current_number_expert] += 1

	return 0

# c экспертом
# определение оптимальной руки с предсказанием
def number_of_arms_f(t):
	global current_number
	max_temp = mean_win[0] / num_of_games[0] + math.sqrt(2 * math.log(t) / num_of_games[0]) + forecast_vec[0] / t * math.exp(-abs(mean_win[0] / num_of_games[0] - forecast_vec[0] / t))
	max_temp_index = 0
	for i in range(n):
		temp[i] = mean_win[i] / num_of_games[i] + math.sqrt(2 * math.log(t) / num_of_games[i]) + forecast_vec[i] / t * math.exp(-abs(mean_win[i] / num_of_games[i] - forecast_vec[i] / t))
		if temp[i] > max_temp:
			max_temp = temp[i]
			max_temp_index = i
	current_number = max_temp_index
	return 0

# пересохраняем вектор сожаления
mean_regret_vec_classic = [0 for i in range(time_finish)]  # вектор сожаления для случая без эксперта
for i in range(time_finish):
	mean_regret_vec_classic[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертами (ucb)
sum_of_reward_f = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]
current_forecast_matr = [[0 for i in range(n)] for j in range(m)]
forecast_matr = [[0 for i in range(n)] for j in range(m)]
temp_expert = [0 for i in range(m)]
current_number_expert = 0
sum_of_choice_expert = [0 for i in range(m)]
mean_win_expert = [0 for i in range(m)]
num_of_games_expert = [0 for i in range(m)]

num_of_choice_accur = [[0 for i in range(n)] for j in range(m)]
sum_num_of_choice_accur = [[0 for i in range(n)] for j in range(m)]


while launch_count < launch_number:
	count = 1

	for i in range(n):
		mean_win[i] = 0
		num_of_games[i] = 1
		forecast_vec[i] = 0
		current_forecast_vec[i] = 0

	for i in range(time_finish):
		regret_vec[i] = 0
		current_number_vector[i] = 0

	for i in range(m):
		for j in range(n):
			current_forecast_matr[i][j] = 0
			forecast_matr[i][j] = 0

	for i in range(m):
		mean_win_expert[i] = 0
		num_of_games_expert[i] = 0

	func_first_win()

	# проинициализировать экспертов
	# выбираем каждого эксперта по разу
	for l in range(m):
		# делаем предсказание
		func_forecast()

		# выбираем l-го эксперта
		choice_expert_initially(l)

		# выбираем действие current_number
		number_of_arms_f(count)

		# запоминаем, какой был выигрыш до этого
		temp_win = mean_win[current_number]

		# играем current_number
		func_win()

		# записываем данные об экспертах
		num_of_games_expert[l] += 1
		mean_win_expert[l] += (mean_win[current_number] - temp_win)
		count += 1

	while count < time_finish:
		# делаются предсказания
		func_forecast()

		# здесь нужно выбрать эксперта
		choice_expert(count)

		# выбран current_number_expert
		# формируем forecast vector
		choice_expert_initially(current_number_expert)

		# после выбора эксперта выбираем действие
		# выбран current_number
		number_of_arms_f(count)

		# запоминаем старый выигрыш
		temp_win = mean_win[current_number]

		# играем это действие
		func_win()

		# запоминаем, что привнес эксперт
		mean_win_expert[current_number_expert] += (mean_win[current_number] - temp_win)
		# print('curr = ', current_number_expert)
		current_number_vector[count] = current_number
		count += 1

	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	for j in range(m):
		sum_of_choice_expert[j] += num_of_games_expert[j]

	# вычисляем вектор сожаления
	func_regret_calculation()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number

for i in range(m):
	sum_of_choice_expert[i] /= launch_number

print('UCB1 choose expert')
print('sum_reward_f = ', sum_reward_f)
print('num_of_expert (ucb) = ', sum_of_choice_expert)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


# пересохраняем вектор сожаления
mean_regret_vec_ucb = [0 for i in range(time_finish)]  # когда выбираем ucb
for i in range(time_finish):
	mean_regret_vec_ucb[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертами (по мере)
sum_of_reward_f = [0 for i in range(n)]
num_of_games_expert = [0 for i in range(m)]
sum_of_choice_expert = [0 for i in range(m)]

launch_count = 0
while launch_count < launch_number:
	count = 1
	for i in range(n):
		mean_win[i] = 0
		num_of_games[i] = 1
		forecast_vec[i] = 0
		current_forecast_vec[i] = 0
	for i in range(time_finish):
		regret_vec[i] = 0
		current_number_vector[i] = 0
	for i in range(m):
		for j in range(n):
			current_forecast_matr[i][j] = 0
			forecast_matr[i][j] = 0

	for i in range(m):
		num_of_games_expert[i] = 0

	func_first_win()

	while count < time_finish:
		func_forecast()
		calc_vec_mera(count)

		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1

	for j in range(m):
		sum_of_choice_expert[j] += num_of_games_expert[j]

	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number

for i in range(m):
	sum_of_choice_expert[i] /= launch_number

print('------mera--------')
print('sum_reward_f = ', sum_reward_f)
print('num_of_expert (mera) = ', sum_of_choice_expert)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number

print('\n')


mean_regret_vec_mera = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_mera[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертами метод 2
# sum_of_reward_f = [0 for i in range(n)]


launch_count = 0
while launch_count < launch_number:
	count = 1
	for i in range(n):
		mean_win[i] = 0
		num_of_games[i] = 1
		forecast_vec[i] = 0
		current_forecast_vec[i] = 0
	for i in range(time_finish):
		regret_vec[i] = 0
		current_number_vector[i] = 0
	for i in range(m):
		for j in range(n):
			current_forecast_matr[i][j] = 0
			forecast_matr[i][j] = 0
			num_of_choice_accur[i][j] = 0


	func_first_win()

	while count < time_finish:
		func_forecast()
		calc_vec_best_accr(count)
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1

	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	for i in range(m):
		for j in range(n):
			sum_num_of_choice_accur[i][j] += num_of_choice_accur[i][j]

	# вычисляем вектор сожаления
	func_regret_calculation()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]


	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number

for i in range(m):
	for j in range(n):
		sum_num_of_choice_accur[i][j] /= launch_number

print('abs accuracy')
print('sum_reward_f = ', sum_reward_f)
print('matrix choice = ', sum_num_of_choice_accur)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number

print('\n')


# вывод графика
time = [i for i in range(time_finish)]


#  для вывода
p1_string = [str(x) for x in p1]
p1_string = ', '.join(p1_string)

p2_string = [str(x) for x in p2]
p2_string = ', '.join(p2_string)


plt.figure(1)
plt.plot(time, mean_regret_vec_classic, linestyle='--', label='UCB1')
plt.plot(time, mean_regret_vec_mera, 'green', label='Method 1')
plt.plot(time, mean_regret_vec, 'r', label='Method 2')
plt.plot(time, mean_regret_vec_ucb, linestyle='-', color='magenta', label='Method 3')
plt.title('The total expected regret (m experts)', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('regret', fontsize=16)
plt.legend(loc='upper left', prop={'size': 12})
plt.grid(True)
plt.show()
