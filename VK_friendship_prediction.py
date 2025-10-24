import vk_api
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class VKGraphAnalyzer:
    def __init__(self, token):

        self.vk = vk_api.VkApi(token=token)
        self.vk_api = self.vk.get_api()
        self.graph = nx.Graph()
        self.user_data = {}

    def get_friends_with_details(self, user_id):  # Получение моих друзей с полями "образование", "пол", "город"
        try:
            friends = self.vk_api.friends.get(
                user_id=user_id,
                fields='education,sex,city'
            )
            time.sleep(0.5)  # задержка чтобы не получить [9] Flood control
            return friends['items']
        except Exception as e:
            print(f"Ошибка при получении друзей {user_id}: {e}")
            return []

    def get_user_info(self, user_id):  # получение информации о пользователе
        try:
            user_info = self.vk_api.users.get(
                user_ids=user_id,
                fields='education,sex,city'
            )[0]
            time.sleep(0.5)
            return {
                'sex': user_info.get('sex', 0),
                'city_id': user_info.get('city', {}).get('id', 0),
                'city_title': user_info.get('city', {}).get('title', ''),
                'university': user_info.get('university', 0)
            }
        except Exception as e:
            print(f"Для id {user_id} vk вернул: {e}")
            return None

    def build_graph_simple(self, initial_user_ids):  # граф: мои друзья + их друзья
        print("\n--- Получение данных о моих друзьях ---")
        for user_id in initial_user_ids:
            print(f"Получение данных об id {user_id}")
            user_info = self.get_user_info(user_id)
            if user_info:
                self.user_data[user_id] = user_info
                print(f"Данные сохранены: {user_info}")

        print("\n--- Получение данных о друзьях моих друзей ---")
        all_friends = set()

        for user_id in initial_user_ids:
            if user_id not in self.user_data:
                continue

            print(f"\nПолучение друзей id {user_id}")
            friends = self.get_friends_with_details(user_id)
            print(f"Найдено друзей: {len(friends)}")

            for friend in friends:
                friend_id = friend['id']
                all_friends.add(friend_id)

                if friend_id not in self.user_data:
                    self.user_data[friend_id] = {
                        'sex': friend.get('sex', 0),
                        'city_id': friend.get('city', {}).get('id', 0),
                        'city_title': friend.get('city', {}).get('title', ''),
                        'university': friend.get('university', 0)
                    }

                self.graph.add_edge(user_id, friend_id)  # добавление ребра между моим другом и другом моего друга

        print(f"\nВсего уникальных друзей: {len(all_friends)}")  # без повторяющихся

        print("\n--- Получение связей между друзьями ---")
        friends_list = list(all_friends)
        for i, friend_id in enumerate(friends_list):
            if i % 50 == 0:
                print(f"Обработано {i}/{len(friends_list)} друзей")

            friend_friends = self.get_friends_with_details(friend_id)  # получение друзей этого друга

            for ff in friend_friends:  # проверка, есть ли среди них другие наши друзья
                ff_id = ff['id']
                if ff_id in all_friends and ff_id != friend_id:
                    self.graph.add_edge(friend_id, ff_id)  # добавление ребра между друзьями

        print(f"\n{'-' * 50}")
        print(f"Граф построен.")
        print(f"Узлов: {self.graph.number_of_nodes()}")
        print(f"Рёбер: {self.graph.number_of_edges()}")
        print(f"Пользователей с данными: {len(self.user_data)}")
        print(f"{'-' * 50}\n")

        return self.graph

    def calculate_centralities(self, user_ids):
        betweenness = nx.betweenness_centrality(self.graph)  # центральность по посредничеству

        closeness = nx.closeness_centrality(self.graph)  # центральность по близости

        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)  # центральность собственного вектора
        except:
            eigenvector = nx.pagerank(self.graph)

        results = {}
        for user_id in user_ids:
            if user_id in self.graph:
                results[user_id] = {
                    'betweenness': betweenness.get(user_id, 0),
                    'closeness': closeness.get(user_id, 0),
                    'eigenvector': eigenvector.get(user_id, 0)
                }
                print(f"\nid \033[34m{user_id}:\033[0m")
                print(f"Посредничество (betweenness): {results[user_id]['betweenness']:.4f}")
                print(f"Близость (closeness): {results[user_id]['closeness']:.4f}")
                print(f"Собственный вектор (eigenvector): {results[user_id]['eigenvector']:.4f}")

        return results

    def prepare_friendship_dataset(self):
        print("\n" + "-" * 50)
        print(f"\033[32mПодготовка датасета для модели\033[0m")
        print("-" * 50)
        print(f"Узлов в графе: {self.graph.number_of_nodes()}")
        print(f"Рёбер в графе: {self.graph.number_of_edges()}")
        print(f"Пользователей в user_data: {len(self.user_data)}")

        # Проверяем, какие узлы есть в графе, но нет в user_data
        missing = set(self.graph.nodes()) - set(self.user_data.keys())  # проверка узло, которые есть в графе, но нет в user_data
        if missing:
            print(f"{len(missing)} узлов в графе без данных")
            print(list(missing)[:5])

        data = []
        nodes = list(self.graph.nodes())

        edges_processed = 0  # позитивные примеры (существующие связи)
        edges_skipped = 0
        for u, v in self.graph.edges():
            if u in self.user_data and v in self.user_data:
                features = self._extract_features(u, v)
                data.append(features + [1])  # 1 - друзья
                edges_processed += 1
            else:
                edges_skipped += 1
                if edges_skipped <= 5:  # первые 5 примеров
                    print(f"Пропущено ребро: {u} - {v}")
                    if u not in self.user_data:
                        print(f"Нет данных о {u}")
                    if v not in self.user_data:
                        print(f"Нет данных о {v}")

        print(f"\nОбработано рёбер: {edges_processed}, пропущено: {edges_skipped}")

        positive_count = len(data)  # негативные примеры (случайные пары без связи)
        negative_count = 0

        nodes_with_data = [n for n in nodes if n in self.user_data]
        print(f"Узлов с данными: {len(nodes_with_data)}")

        if len(nodes_with_data) < 2:
            print("Недостаточно узлов с данными")
            return pd.DataFrame(), pd.Series()

        attempts = 0
        max_attempts = positive_count * 10

        while negative_count < positive_count and attempts < max_attempts:
            u = np.random.choice(nodes_with_data)
            v = np.random.choice(nodes_with_data)
            attempts += 1

            if u != v and not self.graph.has_edge(u, v):
                features = self._extract_features(u, v)
                data.append(features + [0])  # 0 - не друзья
                negative_count += 1

        df = pd.DataFrame(data, columns=[
            'same_sex', 'same_city', 'same_university',
            'common_friends', 'degree_u', 'degree_v',
            'is_friend'
        ])

        X = df.drop('is_friend', axis=1)
        y = df['is_friend']

        print(f"\n{len(X)} примеров в датасете")
        print(f"Друзья: {positive_count}, Не друзья: {negative_count}")
        return X, y

    def _extract_features(self, u, v):
        u_data = self.user_data[u]
        v_data = self.user_data[v]

        same_sex = int(u_data['sex'] == v_data['sex'] and u_data['sex'] != 0)  # совпадает ли пол пользователей. 1 - да, 0 - нет
        same_city = int(u_data['city_id'] == v_data['city_id'] and u_data['city_id'] != 0)  # совпадает ли город пользователей. 1 - да, 0 - нет
        same_university = int(u_data['university'] == v_data['university'] and u_data['university'] != 0)  # совпадает ли вуз пользователей. 1 - да, 0 - нет

        u_friends = set(self.graph.neighbors(u))  # все друзья пользователя u
        v_friends = set(self.graph.neighbors(v))  # все друзья пользователя v
        common_friends = len(u_friends.intersection(v_friends))  # количество общих друзей у пары

        degree_u = self.graph.degree(u)  # степень узла пользователя u (число его друзей в графе)
        degree_v = self.graph.degree(v)  # степень узла пользователя v (число его друзей в графе)

        return [same_sex, same_city, same_university, common_friends, degree_u, degree_v]

    def train_friendship_model(self, X, y):  # обучение модели предсказания дружбы
        print("\n" + "-" * 50 )
        print(f"\033[32mОбучение модели\033[0m")
        print("-" * 50)

        if len(X) == 0:
            print("Датасет пустой")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Результаты на тестовой выборке:")
        print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred, target_names=['Не друзья', 'Друзья']))

        feature_importance = pd.DataFrame({  # важность признаков
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n\033[33m           Важность признаков:\033[0m")
        print(feature_importance)

        return model

    def predict_friendship(self, model, user1_id, user2_id): # предсказание, будут ли два пользователя друзьями
        if user1_id not in self.user_data or user2_id not in self.user_data:
            return None

        features = self._extract_features(user1_id, user2_id)
        X = pd.DataFrame([features], columns=[
            'same_sex', 'same_city', 'same_university',
            'common_friends', 'degree_u', 'degree_v'
        ])

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return {
            'prediction': bool(prediction),
            'probability': probability
        }

    def save_data(self, filename_prefix='vk_graph'):
        with open(f'{filename_prefix}_graph.pkl', 'wb') as f:
            pickle.dump(self.graph, f)
        with open(f'{filename_prefix}_user_data.pkl', 'wb') as f:
            pickle.dump(self.user_data, f)
        print(f"\nДанные сохранены: {filename_prefix}_*.pkl")

    def load_data(self, filename_prefix='vk_graph'):
        with open(f'{filename_prefix}_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        with open(f'{filename_prefix}_user_data.pkl', 'rb') as f:
            self.user_data = pickle.load(f)
        print(f"\nДанные уже загружены: {filename_prefix}_*.pkl")

    def visualize_graph(self, initial_user_ids=None, show_labels=False, save_file='graph.png'):
        print(f"\nУзлов: {self.graph.number_of_nodes()}, Рёбер: {self.graph.number_of_edges()}")

        plt.figure(figsize=(16, 12))
        if self.graph.number_of_nodes() < 100:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=1, iterations=20)

        node_colors = []
        node_sizes = []

        for node in self.graph.nodes():
            if initial_user_ids and node in initial_user_ids:
                node_colors.append('#FF0000')
                node_sizes.append(500)
            else:
                # Остальные - по полу
                if node in self.user_data:
                    sex = self.user_data[node]['sex']
                    if sex == 1:  # женщины
                        node_colors.append('#FF69B4')
                    elif sex == 2:  # мужчины
                        node_colors.append('#4169E1')
                    else:
                        node_colors.append('#808080')
                else:
                    node_colors.append('#808080')

                degree = self.graph.degree(node)
                node_sizes.append(max(20, min(200, degree * 2)))

        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            width=0.5,
            edge_color='gray'
        )

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )

        if show_labels and self.graph.number_of_nodes() < 50:
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=6,
                font_color='black'
            )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF0000', label='Мои друзья'),
            Patch(facecolor='#FF69B4', label='Женщины (друзья моих друзей)'),
            Patch(facecolor='#4169E1', label='Мужчины (друзья моих друзей)'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        plt.title(
            f'Граф связей VK\n{self.graph.number_of_nodes()} узлов, {self.graph.number_of_edges()} рёбер',
            fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nГраф сохранён: {save_file}")
        plt.show()

        return pos

    def visualize_centrality(self, centralities, metric='betweenness', save_file='centrality.png'):
        if metric == 'betweenness':  # вычисление центральностей
            all_centrality = nx.betweenness_centrality(self.graph)
            title = 'Посредничество'
        elif metric == 'closeness':
            all_centrality = nx.closeness_centrality(self.graph)
            title = 'Близость'
        elif metric == 'eigenvector':
            try:
                all_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                all_centrality = nx.pagerank(self.graph)
            title = 'Собственный вектор'

        plt.figure(figsize=(16, 12))

        if self.graph.number_of_nodes() < 100:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=1, iterations=20)

        node_colors = [all_centrality.get(node, 0) for node in self.graph.nodes()]

        node_sizes = [max(50, all_centrality.get(node, 0) * 3000) for node in self.graph.nodes()]

        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, width=0.5, edge_color='gray')

        nodes = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=cm.Reds,
            vmin=0,
            vmax=max(node_colors),
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        plt.colorbar(nodes, label=f'{metric} centrality')

        plt.title(f'{title}\n{self.graph.number_of_nodes()} узлов, {self.graph.number_of_edges()} рёбер',
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(save_file, dpi=300, bbox_inches='tight')

        plt.show()

        # Топ-10 по центральности
        top_nodes = sorted(all_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nТоп-10 узлов по \033[35m{metric}\033[0m:")
        for i, (node, value) in enumerate(top_nodes, 1):
            print(f"{i}. ID \033[34m{node}\033[0m: {value:.4f}")

        return all_centrality


if __name__ == "__main__":
    TOKEN = ""
    GROUP_MEMBERS = [192065628, 353731908, 387361899]

    analyzer = VKGraphAnalyzer(TOKEN)

    BUILD_NEW = False  # True - построить новый, False - загрузить сохранённый

    if BUILD_NEW:
        print("-" * 50)
        print("Построение графа")
        print("-" * 50)
        analyzer.build_graph_simple(GROUP_MEMBERS)
        analyzer.save_data('my_vk_graph')
    else:
        analyzer.load_data('my_vk_graph')

    print("\n" + "-" * 50)
    print(f"\033[32mВычисление центральностей\033[0m")
    print("-" * 50)
    centralities = analyzer.calculate_centralities(GROUP_MEMBERS)

    X, y = analyzer.prepare_friendship_dataset()

    if len(X) > 0:
        model = analyzer.train_friendship_model(X, y)  # обучение модели

        if model:
            with open('friendship_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print("\nМодель сохранена: friendship_model.pkl")

            # if len(list(analyzer.graph.nodes())) >= 2:  # пример предсказания
            #     nodes = list(analyzer.graph.nodes())
            #     user1, user2 = nodes[0], nodes[1]
            #     result = analyzer.predict_friendship(model, user1, user2)
            #     if result:
            #         print(f"\n\033[32mПример предсказания для пары\033[0m (\033[34m{user1}, {user2}\033[0m):")
            #         print(f"Будут друзьями: {result['prediction']}")
            #         print(f"Вероятность: {result['probability']:.2%}")

            if len(list(analyzer.graph.nodes())) >= 4:
                nodes = list(analyzer.graph.nodes())

                print("\n" + "-" * 50)
                print(f"\033[32mПредсказания для трех случайных пар\033[0m")
                print("-" * 50)
                for i in range(3):
                    user1, user2 = np.random.choice(nodes, size=2, replace=False)
                    result = analyzer.predict_friendship(model, user1, user2)
                    if result:
                        print(f"\nПара {i + 1}: (\033[34m{user1}, {user2}\033[0m):")
                        print(f"Будут друзьями: {result['prediction']}")
                        print(f"Вероятность: {result['probability']:.2%}")

    else:
        print("\nДатасет пустой")

    analyzer.visualize_graph(  # визуализация общего графа
        initial_user_ids=GROUP_MEMBERS,
        show_labels=False,
        save_file='full_graph.png'
    )

    if len(X) > 0:  # визуализации центральностей
        analyzer.visualize_centrality(
            centralities,
            metric='betweenness',
            save_file='betweenness.png'
        )

        analyzer.visualize_centrality(
            centralities,
            metric='closeness',
            save_file='closeness.png'
        )

        analyzer.visualize_centrality(
            centralities,
            metric='eigenvector',
            save_file='eigenvector.png'
        )
