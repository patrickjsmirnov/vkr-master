# import random
import math
# import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# действия
p1 = [0.3, 0.45, 0.5, 0.47, 0.1]

# e2
p2 = [0.1, 0.1, 0.6, 0.1, 0.1]
n = len(p1)


# определение лучшей руки
def search_best_arm():
	index_maximum, maximum = 0, p1[0]
	for i in range(n):
		if p1[i] > maximum:
			maximum = p1[i]
			index_maximum = i
	return index_maximum

#  параметры запуска
launch_number = 1
time_finish = 1000
index_best_arm = search_best_arm()


def func_win():
	mean_win[current_number] += bernoulli.rvs(p1[current_number])
	num_of_games[current_number] += 1
	return 0


def func_first_win():
	for i in range(n):
		mean_win[i] = bernoulli.rvs(p1[i])
	return 0


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
	number_of_games[current_number] += 1
	return 0


def func_regret_calculation():
	for i in range(time_finish):
		j, temp_var = 0, 0
		while j < i:
			temp_var += p1[current_number_vector[j]]
			j += 1
		regret_vec[i] = i * p1[index_best_arm] - temp_var
	return 0


# вычисляем доля оптимальных (пока не используется)
def func_per_of_optimal():
	temp_vec = [0 for k in range(time_finish)]
	i = 1
	while i < time_finish:
		if current_number_vector[i] == index_best_arm:
			temp_vec[i] = 1
		j = 0
		s = 0
		while j <= i:
			s += temp_vec[j]
			j += 1
		per_of_optimal[i] = s / i
		i += 1

	return 0

#  без эксперта
sum_of_reward = [0 for i in range(n)]
mean_regret_vec = [0 for i in range(time_finish)]
number_of_games = [0 for i in range(n)]
upper_bound_vector = [0 for i in range(time_finish)]
launch_count = 0
regret_vec = [0 for i in range(time_finish)]
mean_win = [0 for i in range(n)]
num_of_games = [1 for i in range(n)]
current_number_vector = [0 for i in range(time_finish)]
temp = [0 for i in range(n)]

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

	func_per_of_optimal()
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
	number_of_games[i] /= launch_number


print('sum_reward = ', sum_reward)
print('curr_vec = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


# блок с экспертом
#  функция предсказания
def func_forecast():
	for i in range(n):
		# предсказание на текущем шаге
		current_forecast_vec[i] = bernoulli.rvs(p2[i])
		# суммарное предсказание
		forecast_vec[i] += current_forecast_vec[i]
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
	number_of_games[current_number] += 1
	return 0

# пересохраняем вектор сожаления
mean_regret_vec_classic = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_classic[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертом
sum_of_reward_f = [0 for i in range(n)]
number_of_games = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]

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

	func_first_win()

	while count < time_finish:
		func_forecast()
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1


	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()
	func_per_of_optimal()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number
	number_of_games[i] /= launch_number

print('sum_reward_f = ', sum_reward_f)
print('curr_vec_f = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


print('\n')

# второй запуск

p3 = p2
p2 = [0.3, 0.45, 0.5, 0.47, 0.1]

# еще раз с экспертом
# пересохраняем вектор сожаления
mean_regret_vec_f1 = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_f1[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертом
sum_of_reward_f = [0 for i in range(n)]
number_of_games = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]

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

	func_first_win()

	while count < time_finish:
		func_forecast()
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1

	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()
	func_per_of_optimal()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number
	number_of_games[i] /= launch_number

print('sum_reward_f = ', sum_reward_f)
print('curr_vec_f = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number

#  третий запуск

p4 = p2
p2 = [0.1, 0.1, 0.1, 0.1, 0.1]

# еще раз с экспертом
# пересохраняем вектор сожаления
mean_regret_vec_f2 = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_f2[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертом
sum_of_reward_f = [0 for i in range(n)]
number_of_games = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]

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

	func_first_win()

	while count < time_finish:
		func_forecast()
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1


	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()
	func_per_of_optimal()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number
	number_of_games[i] /= launch_number

print('sum_reward_f = ', sum_reward_f)
print('curr_vec_f = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


#четвертый запуск

p5 = p2
p2 = [0.2, 0.3, 0.45, 0.32, 0.05]

# еще раз с экспертом
# пересохраняем вектор сожаления
mean_regret_vec_f3 = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_f3[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертом
sum_of_reward_f = [0 for i in range(n)]
number_of_games = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]

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

	func_first_win()

	while count < time_finish:
		func_forecast()
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1


	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()
	func_per_of_optimal()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number
	number_of_games[i] /= launch_number

print('sum_reward_f = ', sum_reward_f)
print('curr_vec_f = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number

#  пятый  запуск
p6 = p2
p2 = [0.5, 0.45, 0.1, 0.5, 0.7]

# еще раз с экспертом
# пересохраняем вектор сожаления
mean_regret_vec_f4 = [0 for i in range(time_finish)]
for i in range(time_finish):
	mean_regret_vec_f4[i] = mean_regret_vec[i]
	mean_regret_vec[i] = 0


# с экспертом
sum_of_reward_f = [0 for i in range(n)]
sum_per_of_optimal_f = [0 for i in range(time_finish)]
per_of_optimal = [0 for i in range(time_finish)]
number_of_games = [0 for i in range(n)]
launch_count = 0
forecast_vec = [0 for i in range(n)]
current_forecast_vec = [0 for i in range(n)]

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
		per_of_optimal[i] = 0

	func_first_win()

	while count < time_finish:
		func_forecast()
		number_of_arms_f(count)
		func_win()
		current_number_vector[count] = current_number
		count += 1

	for i in range(n):
		sum_of_reward_f[i] += mean_win[i]

	# вычисляем вектор сожаления
	func_regret_calculation()
	func_per_of_optimal()

	for i in range(time_finish):
		mean_regret_vec[i] += regret_vec[i]
		sum_per_of_optimal_f[i] += per_of_optimal[i]

	launch_count += 1

print('\n')

# суммарный выигрыш
sum_reward_f = 0
for i in range(n):
	sum_reward_f += sum_of_reward_f[i] / launch_number
	number_of_games[i] /= launch_number
	sum_per_of_optimal_f[i] /= launch_number

print('sum_reward_f = ', sum_reward_f)
print('curr_vec_f = ', number_of_games)

for i in range(time_finish):
	mean_regret_vec[i] /= launch_number


def upper_bound():
	delta = [0 for i in range(n)]
	temp2 = 0
	for i in range(n):
		if p1[i] < p1[index_best_arm]:
			delta[i] = p1[index_best_arm] - p1[i]
		temp2 += delta[i]

	i = 1
	while i < time_finish:
		temp1 = 0
		for k in range(n):
			if delta[k] != 0:
				temp1 += 8 * math.log(i) / delta[k]
		upper_bound_vector[i] = temp1 + (1 + math.pi * math.pi / 3) * temp2
		i += 1

	return 0


time = [i for i in range(time_finish)]
upper_bound()

#  для вывода
p1_string = [str(x) for x in p1]
p1_string = ', '.join(p1_string)

p2_string = [str(x) for x in p2]
p2_string = ', '.join(p2_string)

p3_string = [str(x) for x in p3]
p3_string = ', '.join(p3_string)

p4_string = [str(x) for x in p4]
p4_string = ', '.join(p4_string)

p5_string = [str(x) for x in p5]
p5_string = ', '.join(p5_string)

p6_string = [str(x) for x in p6]
p6_string = ', '.join(p6_string)


plt.figure(1)
plt.plot(time, mean_regret_vec_classic, linestyle='--', label='UCB1 [' + p1_string + ']')
plt.plot(time, mean_regret_vec, linestyle=':', color='green', label='e1 = [' + p2_string + ']')
plt.plot(time, mean_regret_vec_f1,linestyle='-', color='red', label='e2 = [' + p3_string + ']')
plt.plot(time, mean_regret_vec_f2,linestyle='-', marker='o', markersize=5, markevery=300, color='black', label='e3 = [' + p4_string + ']')
plt.plot(time, mean_regret_vec_f3,linestyle='-', marker='v', markersize=5, markevery=300, color='magenta', label='e4 = [' + p5_string + ']')
plt.plot(time, mean_regret_vec_f4,linestyle='-', marker='s', markersize=5, markevery=300, color='cyan', label='e5 = [' + p6_string + ']')
plt.title('The total expected regret (UCB1 Bernoulli)', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('regret', fontsize=16)
plt.legend(loc='upper left', prop={'size': 12})
plt.grid(True)
plt.show()
