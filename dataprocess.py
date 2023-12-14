import random
from datetime import datetime
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from tqdm import tqdm
from tqdm import trange
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from collections import Counter
import multiprocessing
pd.set_option('display.max_columns', None)


def FileProc(srcroot, csv):
    path = os.path.join(srcroot, csv) 
    df = pd.read_csv(path)
    version = str(srcroot[-4:])
    columns = df.columns.values
    print("\n正在处理" + path+"...\n文件行数：", len(df), columns)
    activities_list = []
    Type = csv[:-4]
    if Type == 'device':
        del_col = ['id', 'file_tree']
        for c in del_col:
            if c in columns:
                del df[c]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            activity = {
                'type': 'device',
                # 'id' : row['id'],
                'date': pd.to_datetime(row['date']),
                'user': row['user'],
                'host': row['pc'],
                # 'file_tree' : row['file_tree'],
                'activity': row['activity'] # connect or disconnect
            }
            activities_list.append(activity)
    elif Type == 'file':
        del_col = ['id', 'content', 'filename', 'to_removable_media', 'from_removable_media']
        for c in del_col:
            if c in columns:
                del df[c]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            activity = {
                'type' : 'file',
                # 'id' : row['id'],
                'date' : pd.to_datetime(row['date']),
                'user' : row['user'],
                'host' : row['pc'],
                # 'file' : row['filename'],
                'activity' :row['activity'] if 'activity' in columns else 'open',
                # 'to_removable_media' : row['to_removable_media'],
                # 'from_removable_media': row['from_removable_media'],
                # 'content': row['content']
            }
            activities_list.append(activity)
    elif Type == 'http':
        del_col = ['id', 'url', 'content']
        for c in del_col:
            if c in columns:
                del df[c]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            activity = {
                'type': 'http',
                # 'id': row['id'],
                'date': pd.to_datetime(row['date']),
                'user': row['user'],
                'host': row['pc'],
                # 'url': row['url'].split(' ')[0],
                'activity' : 'visit',
                # 'content_list' : row['content'].split(' ')[1:]
            }
            activities_list.append(activity)
    elif Type == 'logon':
        del_col = ['id']
        for c in del_col:
            if c in columns:
                del df[c]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            activity = {
                'type': 'logon',
                # 'id': row['id'],
                'date': pd.to_datetime(row['date']),
                'user': row['user'],
                'host': row['pc'],
                'activity': row['activity']
            }
            activities_list.append(activity)
    elif Type == 'email':
        del_col = ['to' 'cc' 'bcc' 'from' 'size' 'attachments', 'content']
        for c in del_col:
            if c in columns:
                del df[c]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            activity = {
                'type': 'logon',
                # 'id': row['id'],
                'date': pd.to_datetime(row['date']),
                'user': row['user'],
                'host': row['pc'],
                'activity': 'send'
            }
            activities_list.append(activity)
    return activities_list


def Convergence(srcroot, root):
    activities_list = []
    is_cached = True
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'device_list.pkl')):
        Device = FileProc(srcroot, "device.csv")
        with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'device_list.pkl'), 'wb') as f:
            pickle.dump(Device, f)
        is_cached = False
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'file_list.pkl')):
        File = FileProc(srcroot, "file.csv")
        with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'file_list.pkl'), 'wb') as f:
            pickle.dump(File, f)
        is_cached = False
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_list.pkl')):
        Http = FileProc(srcroot, "http.csv")
        with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_list.pkl'), 'wb') as f:
            pickle.dump(Http, f)
        is_cached = False
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'logon_list.pkl')):
        Logon = FileProc(srcroot, "logon.csv")
        with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'logon_list.pkl'), 'wb') as f:
            pickle.dump(Logon, f)
        is_cached = False
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'email_list.pkl')):
        Email = FileProc(srcroot, "email.csv") # TODO r6.2
        with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'email_list.pkl'), 'wb') as f:
            pickle.dump(Email, f)
        is_cached = False
    if is_cached == True:
        Device = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'device_list.pkl'), 'rb'))
        File = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'file_list.pkl'), 'rb'))
        Http = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_list.pkl'), 'rb'))
        Logon = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'logon_list.pkl'), 'rb'))
        Email = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'email_list.pkl'), 'rb'))
    activities_list.extend(Device)
    activities_list.extend(File)
    activities_list.extend(Http)
    activities_list.extend(Logon)
    activities_list.extend(Email)
    sorted_activities_list = sorted(activities_list, key=lambda x: (x.__getitem__('user'), x.__getitem__('date')))
    with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'sorted_activities_list.pkl'), 'wb') as f:
        pickle.dump(sorted_activities_list, f)
    return sorted_activities_list

def build_users_dict(srcroot, root):
    cached_cert = os.path.join(root, str(srcroot[-4:]) + '_' + 'dict.pkl')
    if not os.path.isfile(cached_cert):
        if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'sorted_activities_list.pkl')):
            sorted_activities_list = Convergence(srcroot, root)
        else:
            sorted_activities_list = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'sorted_activities_list.pkl'), 'rb'))
        users_dict = {}
        for i in trange(len(sorted_activities_list)):
            cur_user = sorted_activities_list[i]['user']
            if cur_user not in users_dict:
                users_dict[cur_user] = [sorted_activities_list[i]]
            else:
                users_dict[cur_user].append(sorted_activities_list[i])
        with open(cached_cert, "wb") as tf: 
            pickle.dump(users_dict, tf)
    else:
        with open(cached_cert, "rb") as tf:
            users_dict = pickle.load(tf)
    return users_dict

def build_dfall(srcroot, root):
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfall.pkl')):
        users_dict = build_users_dict(srcroot, root)
        users_dict_reduced = OrderedDict()
        for k, v in tqdm(users_dict.items()):
            tmp_activity_list = []
            tmp_date_list = []
            tmp_host_list = []
            for i in v:
                tmp_activity_list.append(i['activity'])
                tmp_date_list.append(i['date'])
                tmp_host_list.append(i['host'])
            users_dict_reduced[k] = {
                'hist_activity': tmp_activity_list,
                'date_list': tmp_date_list,
                'host_list': tmp_host_list
            }
        for k, v in tqdm(users_dict_reduced.items()):
            users_dict_reduced[k]['hist_activity'] = '|'.join(v['hist_activity'])
            users_dict_reduced[k]['host_list'] = '|'.join(v['host_list'])
        user_list = users_dict_reduced.keys()
        activity_list = []
        date_list = []
        host_list = []
        for i in tqdm(users_dict_reduced.values()):
            activity_list.append(i['hist_activity'])
            date_list.append(i['date_list'])
            host_list.append(i['host_list'])
        dfall = pd.DataFrame(data=zip(user_list, activity_list, date_list, host_list),
                               columns=['user_id', 'hist_activity', 'date', 'host'])
        dfall.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfall.pkl'))
    else:
        dfall = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfall.pkl'))
    return dfall



class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_cnt=5, word2idx=None, idx2word=None):
        super().__init__()
        self.min_cnt = min_cnt
        self.word2idx = word2idx if word2idx else dict()
        self.idx2word = idx2word if idx2word else dict()

    def fit(self, x, y=None):
        if not self.word2idx:
            counter = Counter(np.asarray(x).ravel())

            selected_terms = sorted(
                list(filter(lambda x: counter[x] >= self.min_cnt, counter)))

            self.word2idx = dict(
                zip(selected_terms, range(1, len(selected_terms) + 1)))
            self.word2idx['__PAD__'] = 0
            if '__UNKNOWN__' not in self.word2idx:
                self.word2idx['__UNKNOWN__'] = len(self.word2idx)

        if not self.idx2word:
            self.idx2word = {
                index: word for word, index in self.word2idx.items()}

        return self

    def transform(self, x):
        transformed_x = list()
        for term in np.asarray(x).ravel():
            try:
                transformed_x.append(self.word2idx[term])
            except KeyError:
                transformed_x.append(self.word2idx['__UNKNOWN__'])

        return np.asarray(transformed_x, dtype=np.int64)

    def dimension(self):
        return len(self.word2idx)


class SequenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=' ', min_cnt=5, max_len=None,
                 word2idx=None, idx2word=None):
        super().__init__()
        self.sep = sep
        self.min_cnt = min_cnt
        self.max_len = max_len

        self.word2idx = word2idx if word2idx else dict()
        self.idx2word = idx2word if idx2word else dict()

    def fit(self, x, y=None):
        if not self.word2idx:
            counter = Counter()
            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                counter.update(words)
                max_len = max(max_len, len(words))

            if self.max_len is None:
                self.max_len = max_len

            # drop rare words
            words = sorted(list(filter(lambda x: counter[x] >= self.min_cnt, counter)))

            self.word2idx = dict(zip(words, range(1, len(words) + 1)))
            self.word2idx['__PAD__'] = 0
            if '__UNKNOWN__' not in self.word2idx:
                self.word2idx['__UNKNOWN__'] = len(self.word2idx)

        if not self.idx2word:
            self.idx2word = {
                index: word for word, index in self.word2idx.items()}

        if not self.max_len:
            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                max_len = max(max_len, len(words))
            self.max_len = max_len
        return self

    def transform(self, x):
        transformed_x = list()

        for sequence in np.asarray(x).ravel():
            words = list()
            for word in sequence.split(self.sep):
                try:
                    words.append(self.word2idx[word])
                except KeyError:
                    words.append(self.word2idx['__UNKNOWN__'])

            transformed_x.append(
                np.asarray(words[0:self.max_len], dtype=np.int64))

        return np.asarray(transformed_x, dtype=object)

    def dimension(self):
        return len(self.word2idx)

    def max_length(self):
        return self.max_len

def merge_psychometric_and_LDAP(psychometric_root, LDAP_root, dfall):
    psy_df = pd.read_csv(psychometric_root)
    user_psy = {}
    for idx, row in psy_df.iterrows():
        temp_psy = {
            'employee_name': row['employee_name'],
            'user_id': row['user_id'],
            'O': row['O'],
            'C': row['C'],
            'E': row['E'],
            'A': row['A'],
            'N': row['N'],
        }
        if row['user_id'] not in user_psy.keys():
            user_psy[row['user_id']] = temp_psy
    result_lis = []
    user_info = {}
    LDAP_file_list = os.listdir(LDAP_root) 
    for file in LDAP_file_list:
        LDAP_df = pd.read_csv(os.path.join(LDAP_root, file))
        for idx, row in LDAP_df.iterrows():
            temp_LDAP = {
                'employee_name': row['employee_name'],
                'email': row['email'],
                'user_id': row['user_id'],
                'role': row['role'],
                'functional_unit': row['functional_unit'],
                'department': row['department'],
                'team': row['team'],
                'supervisor': row['supervisor']
            }
            if row['user_id'] not in user_info.keys():
                user_info[row['user_id']] = temp_LDAP

    user_psy_list = list(user_psy.values())
    user_info_list = list(user_info.values())
    user_psy_df = pd.DataFrame(user_psy_list)
    del user_psy_df['employee_name']
    user_info_df = pd.DataFrame(user_info_list)
    df = pd.merge(user_info_df, user_psy_df, on='user_id', how='outer')
    for col in ["employee_name", "email", "user_id", "role", 'functional_unit', 'department', 'team',
                'supervisor']:
        df[col] = df[col].astype(str)
    for col in ['O', 'C', 'E', 'A', 'N']:
        df[col] = df[col].astype(int)
    for col in ['functional_unit', 'department', 'team', 'supervisor']:
        df.loc[pd.isnull(df[col]) == True, col] = 'empty'
    dfall = pd.merge(dfall, df, on='user_id', how='outer')
    return dfall


def get_mal_userdata(root, data, usersdf):
    listmaluser = pd.read_csv(os.path.join(root, "answers/insiders.csv"))
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset'] == data.replace("r", "")]
    if data == 'r6.2':
        listmaluser.loc[listmaluser['scenario'] == 4, 'start'] = '02' + listmaluser[listmaluser['scenario'] == 4]['start']
    listmaluser[['start', 'end']] = listmaluser[['start', 'end']].applymap(
        lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
    assert type(usersdf) == pd.core.frame.DataFrame
    usersdf['malscene'] = 0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['user_label'] = 0
    usersdf['acts_labels'] = None
    for i in usersdf.index:
        usersdf.at[i, 'acts_labels'] = len(usersdf.loc[i, 'date']) * [0]

    for i in tqdm(listmaluser.index):
        if data in ['r4.2', 'r5.2']:
            malacts = open(os.path.join(root, f"answers/r{listmaluser['dataset'][i]}-{listmaluser['scenario'][i]}/" +
                                        listmaluser['details'][i]), 'r').read().strip().split("\n")
        else:
            malacts = open(os.path.join(root, "answers/" + listmaluser['details'][i]), 'r').read().replace("\"",'').strip().split("\n")
        malacts = [x.split(',') for x in malacts]
        malLoc = []
        cur_user = listmaluser['user'][i]  # 'AAM0658'
        cur_idx = usersdf.loc[usersdf['user_id'] == cur_user].index[0]  # 4
        dt = usersdf.loc[cur_idx, 'date']
        for m in malacts:
            if m[3] == cur_user:
                malLoc.append(dt.index(pd.to_datetime(m[2])))
            else: 
                usersdf.loc[usersdf.loc[usersdf['user_id'] == m[3]].index[0], 'acts_labels'][
                    usersdf.loc[usersdf.loc[usersdf['user_id'] == m[3]].index[0], 'date'].index(
                        pd.to_datetime(m[2]))] = 1
        usersdf.loc[cur_idx, 'mstart'] = listmaluser['start'][i]
        usersdf.loc[cur_idx, 'mend'] = listmaluser['end'][i]
        usersdf.loc[cur_idx, 'user_label'] = 1
        usersdf.loc[cur_idx, 'malscene'] = listmaluser['scenario'][i]

        for j in malLoc:
            usersdf.loc[cur_idx, 'acts_labels'][j] = 1

    return usersdf


def get_sessions(cur_user, first_sid=0):
    sessions = {}
    open_sessions = {}
    sid = 0
    cur_user_id = cur_user['user_id']
    cur_user['hist_activity'] = cur_user['hist_activity'].split('|')
    cur_user['host'] = cur_user['host'].split('|')
    current_pc = cur_user['host'][0]
    start_time = cur_user['date'][0]
    if cur_user['hist_activity'][0] == 'Logon':
        open_sessions[current_pc] = [current_pc, 1, 0, start_time, start_time, 1, [cur_user['hist_activity'][0]],
                                     [cur_user['date'][0]], [cur_user['acts_labels'][0]]]
    else:
        open_sessions[current_pc] = [current_pc, 2, 0, start_time, start_time, 1, [cur_user['hist_activity'][0]],
                                     [cur_user['date'][0]], [cur_user['acts_labels'][0]]]

    for i in range(1, len(cur_user['date'])):
        current_pc = cur_user['host'][i]
        if current_pc in open_sessions:  # must be already a session with that host
            if cur_user['hist_activity'][i] == 'Logoff':
                open_sessions[current_pc][2] = 1
                open_sessions[current_pc][4] = cur_user['date'][i]
                open_sessions[current_pc][6].append(cur_user['hist_activity'][i])
                open_sessions[current_pc][7].append(cur_user['date'][i])
                open_sessions[current_pc][8].append(cur_user['acts_labels'][i])
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc) + [cur_user_id]
                sid += 1
            elif cur_user['hist_activity'][i] == 'Logon':
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc) + [cur_user_id]
                sid += 1
                open_sessions[current_pc] = [current_pc, 1, 0, cur_user['date'][i], cur_user['date'][i], 1,
                                             [cur_user['hist_activity'][i]], [cur_user['date'][i]],
                                             [cur_user['acts_labels'][i]]]
                if len(open_sessions) > 1: 
                    for k in open_sessions:
                        open_sessions[k][5] += 1
            else:
                open_sessions[current_pc][4] = cur_user['date'][i]
                open_sessions[current_pc][6].append(cur_user['hist_activity'][i])
                open_sessions[current_pc][7].append(cur_user['date'][i])
                open_sessions[current_pc][8].append(cur_user['acts_labels'][i])
        else:
            start_status = 1 if cur_user['hist_activity'][i] == 'Logon' else 2
            open_sessions[current_pc] = [current_pc, start_status, 0, cur_user['date'][i], cur_user['date'][i], 1,
                                         [cur_user['hist_activity'][i]], [cur_user['date'][i]],
                                         [cur_user['acts_labels'][i]]]
            if len(open_sessions) > 1:
                for k in open_sessions:
                    open_sessions[k][5] += 1
    return sessions

def mp_worker(row):
    raise NotImplementedError

def session_spilt(root, srcroot, dfall, identifier=None):
    if identifier:
        save_file = os.path.join(root, str(srcroot[-4:]) + '_' + 'sessions'+str(identifier)+'.pkl')
    else:
        save_file = os.path.join(root, str(srcroot[-4:]) + '_' + 'sessions.pkl')
    if os.path.isfile(save_file):
        sessions = pd.read_pickle(save_file)
    else:
        # # parallel implementation
        with multiprocessing.Pool(6) as pool:
            all_uesr_sessions = tqdm(pool.imap(get_sessions, [row for idx, row in dfall.iterrows()]), total=len(dfall))
            _, pcs, _, _, _, _, _, session_based_action_sequences, session_based_date_sequences, session_based_acts_label_sequences, user_ids = zip(*[x for s in all_uesr_sessions for x in s.values()])
            # pcs: x[1]
            # session_based_action_sequences: x[7]
            # session_based_date_sequences: x[8]
            # session_based_acts_label_sequencesL: x[9]
            # user_ids: x[-1] 即 x[10]
        # # serial implementation
        # session_based_action_sequences = []
        # session_based_date_sequences = []
        # session_based_acts_label_sequences = []
        # user_ids = []
        # pcs = []
        # for i in trange(len(dfall)):
        #     cur_uesr_sessions = get_sessions(dfall.iloc[i])
        #     temp_session_based_action_sequences = [x[7] for x in cur_uesr_sessions.values()]
        #     temp_session_based_date_sequences = [x[8] for x in cur_uesr_sessions.values()]
        #     temp_session_based_acts_label_sequences = [x[9] for x in cur_uesr_sessions.values()]
        #     temp_user_id = [dfall.iloc[i]['user_id']] * len(temp_session_based_action_sequences)
        #     temp_pc = [x[1] for x in cur_uesr_sessions.values()]
        #     session_based_action_sequences.extend(temp_session_based_action_sequences)
        #     session_based_date_sequences.extend(temp_session_based_date_sequences)
        #     session_based_acts_label_sequences.extend(temp_session_based_acts_label_sequences)
        #     user_ids.extend(temp_user_id)
        #     pcs.extend(temp_pc)
        sessions = pd.DataFrame(data=zip(user_ids, pcs, session_based_action_sequences, session_based_date_sequences,
                                         session_based_acts_label_sequences),
                                columns=['user_id', 'host', 'hist_activity', 'date', 'acts_labels'])
        print('sessions占用内存: {:.2f} GB'.format(sessions.memory_usage().sum() / (1024 ** 3)))
        sessions.to_pickle(save_file)
    return sessions

def numerize_dfall(srcroot, root):
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'numeric_session_dfall.pkl')):
        if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'labeled_session_dfall.pkl')):
            dfall = build_dfall(srcroot, root)
            dfall = merge_psychometric_and_LDAP(os.path.join(srcroot, 'psychometric.csv'), os.path.join(srcroot, 'LDAP'), dfall)
            del dfall['employee_name']
            del dfall['email']
            dfall = get_mal_userdata(root, data=str(srcroot[-4:]), usersdf=dfall)
            if srcroot[-4:] == 'r6.2':
                lendf = len(dfall)
                print("dfLen:{}".format(lendf))
                sessions = [None, None, None, None]
                sessions[0] = session_spilt(root, srcroot, dfall[:1000], identifier=1)
                sessions[1] = session_spilt(root, srcroot, dfall[1000:2000], identifier=2)
                sessions[2] = session_spilt(root, srcroot, dfall[2000:3000], identifier=3)
                sessions[3] = session_spilt(root, srcroot, dfall[3000:], identifier=4)
                sessions = pd.concat(sessions, ignore_index=True)
            elif srcroot[-4:] == 'r4.2' or srcroot[-4:] == 'r5.2':
                sessions = session_spilt(root, srcroot, dfall)
            else:
                raise NotImplementedError
            del dfall['hist_activity']
            del dfall['host']
            del dfall['date']
            del dfall['acts_labels']
            del dfall['mstart']
            del dfall['mend']
            dfall = dfall.merge(sessions, on='user_id', how='outer')
            print(max([len(i) for i in list(dfall['hist_activity'])]))
            dfall['hist_activity'] = dfall['hist_activity'].map(lambda x: '|'.join(x))
            dfall['session_label'] = [1 if 1 in i else 0 for i in dfall['acts_labels']]
            dfall['acts_labels'] = dfall['acts_labels'].map(lambda x: np.array(x))
            dfall.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'labeled_session_dfall.pkl'))
        else:
            dfall = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'labeled_session_dfall.pkl'))
        if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'origin_numeric_session_dfall.pkl')):
            encoders = {}
            num_features = ['O', 'C', 'E', 'A', 'N']
            num_pipe = Pipeline(steps=[('impute', SimpleImputer()), ('quantile', QuantileTransformer())])
            print("preprocess number features...")
            dfall[num_features] = num_pipe.fit_transform(dfall[num_features]).astype(np.float32)
            cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
            print("preprocess category features...")
            for col in tqdm(cat_features):
                encoders[col] = CategoryEncoder(min_cnt=1)
                dfall[col] = encoders[col].fit_transform(dfall[col])
            seq_features = ['hist_activity']
            print("preprocess sequence features...")
            for col in tqdm(seq_features):
                encoders[col] = SequenceEncoder(sep="|", min_cnt=1)
                dfall[col] = encoders[col].fit_transform(dfall[col])
            dfall.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'origin_numeric_session_dfall.pkl'))
            with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), "wb") as tf: 
                pickle.dump(encoders, tf)
        else:
            encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
            dfall = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'origin_numeric_session_dfall.pkl'))
        tqdm.pandas(desc='apply')
        print(dfall.iloc[10]['hist_activity'])
        print(len(dfall.iloc[10]['hist_activity']))
        print(len(dfall.iloc[10]['date']))
        dfall['hist_activity'] = dfall[['hist_activity', 'date']].progress_apply(lambda x: np.array(
            [(x['hist_activity'][i] - 1) * 24 + x['date'][i].hour + 1 for i in range(len(x['hist_activity'])) if
             x['hist_activity'][i] > 0]), axis=1)
        print(dfall['hist_activity'][0])
        dfall.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'numeric_session_dfall.pkl'))
    else:
        num_features = ['O', 'C', 'E', 'A', 'N']
        cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
        seq_features = ['hist_activity']
        encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
        dfall = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'numeric_session_dfall.pkl'))
    return num_features, cat_features, seq_features, encoders, dfall

def split_train_val_with_date(srcroot, root, sp='2011-01-01 00:00:00'):
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain.pkl')) or not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval.pkl')):
        if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_reduced_numeric_session_dfall.pkl')):
            num_features, cat_features, seq_features, encoders, dfall = numerize_dfall(srcroot, root)
            dfall = httpreduce(srcroot, root, dfall)
            encoders['hist_activity'].max_len = max([len(i) for i in dfall['hist_activity'].values])
            with open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), "wb") as tf:
                pickle.dump(encoders, tf)
        else:
            num_features = ['O', 'C', 'E', 'A', 'N']
            cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
            seq_features = ['hist_activity']
            encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
            dfall = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_reduced_numeric_session_dfall.pkl'))
        dfall['temp_date'] = dfall['date'].apply(lambda x: x[0])
        dftrain = dfall.loc[dfall['temp_date'] < pd.to_datetime(sp)]
        dfval = dfall.loc[dfall['temp_date'] > pd.to_datetime(sp)]
        del dftrain['temp_date']
        del dfval['temp_date']
        dftrain.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain.pkl'))
        dfval.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval.pkl'))
    else:
        num_features = ['O', 'C', 'E', 'A', 'N']
        cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
        seq_features = ['hist_activity']
        encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
        dftrain = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain.pkl'))
        dfval = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval.pkl'))
    s1 = set(dftrain[dftrain['session_label'] == 1]['user_id'].values)
    s2 = set(dfval[dfval['session_label'] == 1]['user_id'].values)
    return num_features, cat_features, seq_features, encoders, [dftrain, dfval]

def httpreduce(srcroot, root, dfall):
    for idx in tqdm(dfall.index):
        ha = dfall.at[idx, 'hist_activity']
        dt = dfall.at[idx, 'date']
        al = dfall.at[idx, 'acts_labels']
        l = len(ha)
        mask = np.full(l, True)
        for k in range(l-1, 0, -1):
            if ha[k] == ha[k-1] and al[k] != 1:
                mask[k] = False
                dt.pop(k)
        dfall.at[idx, 'hist_activity'] = ha[mask]
        dfall.at[idx, 'acts_labels'] = al[mask]
    dfall.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'http_reduced_numeric_session_dfall.pkl'))
    return dfall

import random
def helper_func(dfdata, do_anomaly=False, do_all=False):
    action_seq_list = []
    target_action_list = []
    target_action_label_list = []
    other_dict = {}
    for idx in tqdm(dfdata.index):
        ha = dfdata.at[idx, 'hist_activity']
        al = dfdata.at[idx, 'acts_labels']
        n = len(ha)
        if 1 in al:
            index_subset = range(1, n)
        elif do_all:
            index_subset = range(1, n)
        else:
            if n <= 1:
                continue
            subset_size = int(n * 0.5)
            if subset_size == 0:
                subset_size = 1
            index_subset = random.sample(range(1, n), subset_size)
        for i in index_subset:
            action_seq_list.append(ha[:i])
            target_action_list.append(ha[i])
            target_action_label_list.append(al[i])
            for k in dfdata.columns:
                if k != 'hist_activity' and k!='acts_labels' and k!='date':
                    if k not in other_dict:
                        other_dict[k] = [dfdata.at[idx, k]]
                    else:
                        other_dict[k].append(dfdata.at[idx, k])
    other_dict['hist_activity'] = action_seq_list
    other_dict['target_action'] = target_action_list
    other_dict['target_action_label'] = target_action_label_list
    dfdata_action = pd.DataFrame(data=other_dict)
    return dfdata_action

def pick_random_elements(total_num, except_x, num_elements):
    lst = np.arange(1, total_num, 1)
    lst = lst[lst != except_x]
    if num_elements >= len(lst):
        return lst
    random_elements = np.array(random.sample(list(lst), num_elements))
    return random_elements

def action_sample(srcroot, root):
    if not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain_action.pkl')) or not os.path.isfile(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval_action_do_all.pkl')):
        if srcroot[-4:] == 'r6.2':
            num_features, cat_features, seq_features, encoders, df = split_train_val_with_date(srcroot, root, sp='2011-03-01 00:00:00')
        else: 
            num_features, cat_features, seq_features, encoders, df = split_train_val_with_date(srcroot, root, sp='2011-01-01 00:00:00')
        dftrain, dfval = df
        dftrain = helper_func(dftrain)
        dftrain.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain_action.pkl'))
        dfval = helper_func(dfval, do_all=True)
        dfval.to_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval_action_do_all.pkl'))
    else:
        num_features = ['O', 'C', 'E', 'A', 'N']
        cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
        seq_features = ['hist_activity']
        encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
        dftrain = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain_action.pkl'))
        dfval = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval_action_do_all.pkl'))
    return num_features, cat_features, seq_features, encoders, [dftrain, dfval]
