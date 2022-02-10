from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from pprint import pprint
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import operator
import argparse
from os import listdir
from os.path import isfile, join

DATA_PATH = '../dataset/'


class PreprocessDataset:
    dataset_name = ''
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    
    def binarize(self):
        X = pickle.load(open(DATA_PATH+self.dataset_name+'_features.p', 'rb'))
        y = pickle.load(open(DATA_PATH+self.dataset_name+'_labels.p', 'rb'))
        
        print(X, X.shape)
        X = np.where(X != 0, 1, 0)
        print(X, X.shape)
        
        print(y, y.shape)
        
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_binary_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_binary_labels.p', 'wb'))
    
    
    def preprocess(self):
        if self.dataset_name == 'purchase_100':
            self.preprocess_purchase()
        elif self.dataset_name == 'location':
            self.preprocess_location()
        elif self.dataset_name == 'compas':
            self.preprocess_compas()
        elif self.dataset_name == 'census':
            #self.preprocess_census()
            self.preprocess_all_census()
        elif self.dataset_name == 'texas_100_v2':
            self.preprocess_texas()
    
    
    def normalizeDataset(X):
        mods = np.linalg.norm(X, axis=1)
        return X / mods[:, np.newaxis]
    
    
    def preprocess_purchase(self):
        IT_NUM = 600
        
        def populate():
            # Note: transactions.csv file can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
            fp = open(DATA_PATH+'transactions.csv')
            cnt, cust_cnt, it_cnt = 0, 0, 0
            items = dict()
            customer = dict()
            last_cust = ''
            for line in fp:
                cnt += 1
                if cnt == 1:
                    continue
                cust = line.split(',')[0]
                it = line.split(',')[3]
                if it not in items and it_cnt < IT_NUM:
                    items[it] = it_cnt
                    it_cnt += 1
                if cust not in customer:
                    customer[cust] = [0]*IT_NUM
                    cust_cnt += 1
                    last_cust = cust
                if cust_cnt > 250000:
                    break
                if it in items:
                    customer[cust][items[it]] = 1
                if cnt % 10000 == 0:
                    print(cnt, cust_cnt, it_cnt)
            del customer[last_cust]
            print(len(customer), len(items))
            
            no_purchase = []
            for key, val in customer.items():
                if 1 not in val:
                    no_purchase.append(key)
            for cus in no_purchase:
                del customer[cus]
            print(len(customer), len(items))
            pickle.dump([customer, items], open(DATA_PATH+'transactions_dump.p', 'wb'))

        populate()
        X = []
        customer, items = pickle.load(open(DATA_PATH+'transactions_dump.p', 'rb'))
        for key, val in customer.items():
            X.append(val)
        X = np.array(X)
        X = self.normalizeDataset(X)
        print(X.shape)
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        y = KMeans(n_clusters=100, random_state=0).fit(X).labels_
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        print(np.unique(y))
    
    
    def preprocess_location(self):
        X = []
        y = []
        with open(DATA_PATH+'bangkok_location') as fp:
            for line in fp:
                line = line.strip().split(',')
                y.append(int(line[0].strip('"')))
                X.append([int(v) for v in line[1:]])
        X = np.matrix(X)
        y = np.array(y)
        print(X.shape, y.shape)
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
    
    
    def preprocess_compas(self):
        X = []
        y = []
        # Note: cox-violent-parsed_filt.csv file can be downloaded from https://www.kaggle.com/danofer/compass
        df = pd.read_csv(DATA_PATH+'cox-violent-parsed_filt.csv')
        df = df[['sex', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_violent_recid']]
        
        # changing the negative COMPAS decile score to zero (this does not alter the meaning)
        df['decile_score'] = [0 if val == -1 else val for val in df['decile_score']]
        
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        #print(df.isnull().any(axis=0))
        
        cat_columns = df.select_dtypes('object').columns
        df[cat_columns] = df[cat_columns].astype('category')
        
        attribute_dict = dict(zip(cat_columns, [dict(zip(df[col].cat.categories, range(len(df[col].cat.categories)))) for col in cat_columns]))
        pprint(attribute_dict)
        
        for col in cat_columns:
            df[col] = df[col].apply(lambda x: attribute_dict[col][x])
        print(df.head())
        
        attribute_idx = dict(zip(cat_columns, [df.columns.get_loc(col) for col in cat_columns]))
        pprint(attribute_idx)
        
        for col in attribute_idx:
            attribute_dict[attribute_idx[col]] = {v: k for k, v in attribute_dict.pop(col).items()}
        pprint(attribute_dict)
        
        y = np.array(df['is_violent_recid'])
        X = np.matrix(df.drop(columns='is_violent_recid'))
        print(X.shape, y.shape)
        
        max_attr_vals = np.max(X, axis=0)
        X = X / max_attr_vals
        max_attr_vals = np.squeeze(np.array(max_attr_vals))
        print(max_attr_vals, max_attr_vals.shape)
        
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        pickle.dump([attribute_idx, attribute_dict, max_attr_vals], open(DATA_PATH+self.dataset_name+'_feature_desc.p', 'wb'))
    
    
    def preprocess_census(self):
        X = []
        y = []
        # Note: psam_p51.csv file can be downloaded from https://www.census.gov/programs-surveys/acs/microdata/access.html
        df = pd.read_csv(DATA_PATH+self.dataset_name+'/psam_p51.csv')
        ''' to make a data set similar to Adult data set, filter rows such that 
            df['AGEP'] > 16 && df['WKHP'] > 0, task : df['PINCP'] > 50k
            'PWGTP' should not be used for model training, its possibly for data sampling
        '''
        print(df)
        df = df[['AGEP', 'COW', 'SCHL', 'MAR', 'RAC1P', 'SEX', 'DREM', 'DPHY', 'DEAR', 'DEYE', 'WKHP', 'WAOB', 'PINCP']]
        df = df[df['AGEP'] > 16]
        df = df[df['WKHP'] > 0]
        for col in ['DREM', 'DPHY', 'DEAR', 'DEYE']:
            df.loc[df[col] == 2, col] = 0
        for col in ['SEX', 'COW', 'SCHL', 'MAR', 'RAC1P', 'WAOB']:
            df[col] -= 1
        df = df.astype('int64')
        print(df)
        #print(df.isnull().any(axis=0))
        print('Number of records with Income > $50K: %d' % sum(df['PINCP'] > 50000))
        # combining Alaskan Native and American Indian, and their combinations into one class
        df.loc[df['RAC1P'] == 3, 'RAC1P'] = 2
        df.loc[df['RAC1P'] == 4, 'RAC1P'] = 2
        # readjusting the number of classes
        for i in range(5, 9):
            df.loc[df['RAC1P'] == i, 'RAC1P'] = i - 2
        
        # creating the categorical attribute dictionary
        attribute_dict = {}
        for col, desc in zip(['DREM', 'DPHY', 'DEAR', 'DEYE'], ['Cognitive Difficulty', 'Ambulatory Difficulty', 'Hearing Difficulty', 'Vision Difficulty']):
            attribute_dict[df.columns.get_loc(col)] = {0: 'No ' + desc, 1: desc}
        attribute_dict[df.columns.get_loc('SEX')] = {0: 'Male', 1: 'Female'}
        attribute_dict[df.columns.get_loc('COW')] = {0: 'Private For-Profit', 1: 'Private Non-Profit', 2: 'Local Govt', 3: 'State Govt', 4: 'Federal Govt', 5: 'Self-Employed Other', 6: 'Self-Employed Own', 7: 'Unpaid Job', 8: 'Unemployed'}
        attribute_dict[df.columns.get_loc('MAR')] = {0: 'Married', 1: 'Widowed', 2: 'Divorced', 3: 'Separated', 4: 'Never married'}
        attribute_dict[df.columns.get_loc('RAC1P')] = {0: 'White', 1: 'Black', 2: 'Native American', 3: 'Asian', 4: 'Pacific Islander', 5: 'Some Other Race', 6: 'Two or More Races'}
        attribute_dict[df.columns.get_loc('WAOB')] = {0: 'US state', 1: 'PR and US Island Areas', 2: 'Latin America', 3: 'Asia', 4: 'Europe', 5: 'Africa', 6: 'Northern America', 7: 'Oceania and at Sea'}
        attribute_dict[df.columns.get_loc('SCHL')] = {0: 'No schooling completed', 1: 'Nursery school, preschool', 2: 'Kindergarten', 3: 'Grade 1', 4: 'Grade 2', 5: 'Grade 3', 6: 'Grade 4', 7: 'Grade 5', 8: 'Grade 6', 9: 'Grade 7', 10: 'Grade 8', 11: 'Grade 9', 12: 'Grade 10', 13: 'Grade 11', 14: '12th grade - no diploma', 15: 'Regular high school diploma', 16: 'GED or alternative credential', 17: 'Some college, but less than 1 year', 18: '1 or more years of college credit, no degree', 19: 'Associates degree', 20: 'Bachelors degree', 21: 'Masters degree', 22: 'Professional degree beyond a bachelors degree', 23: 'Doctorate degree'}
        
        attribute_idx = {col: df.columns.get_loc(col) for col in ['DREM', 'DPHY', 'DEAR', 'DEYE', 'SEX', 'COW', 'MAR', 'RAC1P', 'WAOB', 'SCHL']}
        pprint(attribute_idx)
        pprint(attribute_dict)
        
        y = np.array(df['PINCP'])
        y = np.where(y > 50000, 1, 0)
        X = np.matrix(df.drop(columns='PINCP'))
        print(X.shape, y.shape)
        
        max_attr_vals = np.max(X, axis=0)
        X = X / max_attr_vals
        max_attr_vals = np.squeeze(np.array(max_attr_vals))
        print(max_attr_vals, max_attr_vals.shape)
        
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        pickle.dump([attribute_idx, attribute_dict, max_attr_vals], open(DATA_PATH+self.dataset_name+'_feature_desc.p', 'wb'))
    
    
    def preprocess_all_census(self):
        X = None
        y = None
        path = DATA_PATH+self.dataset_name
        allDataFiles = [f for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.csv'))]
        df_all = None
        #print("All data files")
        #print(allDataFiles)
        #go over all csv files in the census data folder
        for file in allDataFiles:
            # Note: psam_p51.csv file can be downloaded from https://www.census.gov/programs-surveys/acs/microdata/access.html
            df = pd.read_csv(path+"/"+file)
            ''' to make a data set similar to Adult data set, filter rows such that 
                df['AGEP'] > 16 && df['WKHP'] > 0, task : df['PINCP'] > 50k
                'PWGTP' should not be used for model training, its possibly for data sampling
                PS: the ST (state) column is included to denote the geographical region of the data records.
            '''
            df = df[['AGEP', 'COW', 'SCHL', 'MAR', 'RAC1P', 'SEX', 'DREM', 'DPHY', 'DEAR', 'DEYE', 'WKHP', 'WAOB', 'ST', 'PINCP']]
            df = df[df['AGEP'] > 16]
            df = df[df['WKHP'] > 0]
            for col in ['DREM', 'DPHY', 'DEAR', 'DEYE']:
                df[col][df[col] == 2] = 0
            for col in ['SEX', 'COW', 'SCHL', 'MAR', 'RAC1P', 'WAOB']:
                df[col] -= 1
            df = df.astype('int64')
            #print(df)
            #print(df.isnull().any(axis=0))
            print('Number of records with Income > $50K: %d' % sum(df['PINCP'] > 50000))
            # combining Alaskan Native and American Indian, and their combinations into one class
            df['RAC1P'][df['RAC1P'] == 3] = 2
            df['RAC1P'][df['RAC1P'] == 4] = 2
            # readjusting the number of classes
            for i in range(5, 9):
                df['RAC1P'][df['RAC1P'] == i] = i - 2
            
            #concat data in the current file to the aggregated data
            if df_all is not None:
                df_all = pd.concat([df_all,df])
            else:
                df_all = df
        #Remove unused keys for ST attribute (e.g. there is no state for value 3)
        unique_st = np.sort(df_all['ST'].unique())
        for old_val,new_val in zip(unique_st,list(range(len(unique_st)))):
            df_all['ST'][df_all['ST']==old_val] = new_val

        y = np.array(df_all['PINCP'])
        y = np.where(y >= 50000, 1, 0)
        X = np.matrix(df_all.drop(columns='PINCP'))
        
        

        # creating the categorical attribute dictionary
        attribute_dict = {}
        for col, desc in zip(['DREM', 'DPHY', 'DEAR', 'DEYE'], ['Cognitive Difficulty', 'Ambulatory Difficulty', 'Hearing Difficulty', 'Vision Difficulty']):
            attribute_dict[df_all.columns.get_loc(col)] = {0: 'No ' + desc, 1: desc}
        attribute_dict[df_all.columns.get_loc('SEX')] = {0: 'Male', 1: 'Female'}
        attribute_dict[df_all.columns.get_loc('COW')] = {0: 'Private For-Profit', 1: 'Private Non-Profit', 2: 'Local Govt', 3: 'State Govt', 4: 'Federal Govt', 5: 'Self-Employed Other', 6: 'Self-Employed Own', 7: 'Unpaid Job', 8: 'Unemployed'}
        attribute_dict[df_all.columns.get_loc('MAR')] = {0: 'Married', 1: 'Widowed', 2: 'Divorced', 3: 'Separated', 4: 'Never married'}
        attribute_dict[df_all.columns.get_loc('RAC1P')] = {0: 'White', 1: 'Black', 2: 'Native American', 3: 'Asian', 4: 'Pacific Islander', 5: 'Some Other Race', 6: 'Two or More Races'}
        attribute_dict[df_all.columns.get_loc('WAOB')] = {0: 'US state', 1: 'PR and US Island Areas', 2: 'Latin America', 3: 'Asia', 4: 'Europe', 5: 'Africa', 6: 'Northern America', 7: 'Oceania and at Sea'}
        attribute_dict[df_all.columns.get_loc('ST')] = {
            0:'Alabama/AL',
            1:'Alaska/AK',
            2:'Arizona/AZ',
            3:'Arkansas/AR',
            4:'California/CA',
            5:'Colorado/CO',
            6:'Connecticut/CT',
            7:'Delaware/DE',
            8:'District of Columbia/DC',
            9:'Florida/FL',
            10:'Georgia/GA',
            11:'Hawaii/HI',
            12:'Idaho/ID',
            13:'Illinois/IL',
            14:'Indiana/IN',
            15:'Iowa/IA',
            16:'Kansas/KS',
            17:'Kentucky/KY',
            18:'Louisiana/LA',
            19:'Maine/ME',
            20:'Maryland/MD',
            21:'Massachusetts/MA',
            22:'Michigan/MI',
            23:'Minnesota/MN',
            24:'Mississippi/MS',
            25:'Missouri/MO',
            26:'Montana/MT',
            27:'Nebraska/NE',
            28:'Nevada/NV',
            29:'New Hampshire/NH',
            30:'New Jersey/NJ',
            31:'New Mexico/NM',
            32:'New York/NY',
            33:'North Carolina/NC',
            34:'North Dakota/ND',
            35:'Ohio/OH',
            36:'Oklahoma/OK',
            37:'Oregon/OR',
            38:'Pennsylvania/PA',
            39:'Rhode Island/RI',
            40:'South Carolina/SC',
            41:'South Dakota/SD',
            42:'Tennessee/TN',
            43:'Texas/TX',
            44:'Utah/UT',
            45:'Vermont/VT',
            46:'Virginia/VA',
            47:'Washington/WA',
            48:'West Virginia/WV',
            49:'Wisconsin/WI',
            50:'Wyoming/WY',
            51:'Puerto Rico/PR'
            }
        
        attribute_dict[df_all.columns.get_loc('SCHL')] = {0: 'No schooling completed', 1: 'Nursery school, preschool', 2: 'Kindergarten', 3: 'Grade 1', 4: 'Grade 2', 5: 'Grade 3', 6: 'Grade 4', 7: 'Grade 5', 8: 'Grade 6', 9: 'Grade 7', 10: 'Grade 8', 11: 'Grade 9', 12: 'Grade 10', 13: 'Grade 11', 14: '12th grade - no diploma', 15: 'Regular high school diploma', 16: 'GED or alternative credential', 17: 'Some college, but less than 1 year', 18: '1 or more years of college credit, no degree', 19: 'Associates degree', 20: 'Bachelors degree', 21: 'Masters degree', 22: 'Professional degree beyond a bachelors degree', 23: 'Doctorate degree'}
        
        attribute_idx = {col: df_all.columns.get_loc(col) for col in ['DREM', 'DPHY', 'DEAR', 'DEYE', 'SEX', 'COW', 'MAR', 'RAC1P', 'WAOB', 'SCHL', 'ST']}
        pprint(attribute_idx)
        pprint(attribute_dict)
            
        max_attr_vals = np.max(X, axis=0)
        X = X / max_attr_vals
        
        max_attr_vals = np.squeeze(np.array(max_attr_vals))
        print(max_attr_vals, max_attr_vals.shape)
        print(f"Length of y: {len(y)}")
        print(f"Dimension of X: {X.shape}")
        np.savetxt(DATA_PATH+self.dataset_name+"_features.csv", X, delimiter=",")
        np.savetxt(DATA_PATH+self.dataset_name+"_labels.csv", y, delimiter=",")
        print(attribute_idx)
        print(attribute_dict)
        print(max_attr_vals)

        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        pickle.dump([attribute_idx, attribute_dict, max_attr_vals], open(DATA_PATH+self.dataset_name+'_feature_desc.p', 'wb'))
        
    
    def preprocess_texas(self):
        # Note: PUDF_base1q2006_tab.txt file can be downloaded from https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm
        df = pd.read_csv(DATA_PATH+self.dataset_name+'/PUDF_base1q2006_tab.txt', sep='\t')
        df = df[['SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', 'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS', 'PRINC_SURG_PROC_CODE']]
        cat_attrs = ['SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'PAT_STATUS', 'RACE', 'ETHNICITY', 'ADMITTING_DIAGNOSIS', 'PRINC_SURG_PROC_CODE']
        df.loc[df['SEX_CODE'] == 'M', 'SEX_CODE'] = 0
        df.loc[df['SEX_CODE'] == 'F', 'SEX_CODE'] = 1
        df.loc[df['SEX_CODE'] == 'U', 'SEX_CODE'] = None
        for col in cat_attrs:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
            df[col] = df[col].astype('int64')
        df.dropna(inplace=True)
        for col in ['LENGTH_OF_STAY', 'PAT_AGE']:
            df[col] = df[col].astype('int64')
        df['TOTAL_CHARGES'] = df['TOTAL_CHARGES'].astype('float64')
        top_100_surgery = dict(Counter(df['PRINC_SURG_PROC_CODE']))
        top_100_surgery = sorted(top_100_surgery.items(), key=(lambda x: x[1]), reverse=True)[:100]
        top_100_surgery = [surgery[0] for surgery in top_100_surgery]
        for idx in df.index:
            if df['PRINC_SURG_PROC_CODE'][idx] not in top_100_surgery:
                df['PRINC_SURG_PROC_CODE'][idx] = None
        df.dropna(inplace=True)
        df['PRINC_SURG_PROC_CODE'] = df['PRINC_SURG_PROC_CODE'].astype('int64')
        print(df)
        for col in df.columns:
            print(col, len(set(df[col])))
        mapping = {}
        for col in cat_attrs:
            mapping[col] = {b:a for a,b in enumerate(sorted(list(Counter(df[col]).keys())))}
            for key, value in mapping[col].items():
                df.loc[df[col] == key, col] = value
        print(df)
        
        # creating the categorical attribute dictionary
        attribute_dict = {}
        for col in cat_attrs:
            if col == 'SEX_CODE':
                attribute_dict[df.columns.get_loc('SEX_CODE')] = {0: 'Male', 1: 'Female'}
            elif col == 'PRINC_SURG_PROC_CODE':
                continue
            else:
                attribute_dict[df.columns.get_loc(col)] = {val: str(val) for val in mapping[col].values()}
        
        attribute_idx = {col: df.columns.get_loc(col) for col in set(cat_attrs) - {'PRINC_SURG_PROC_CODE'}}
        pprint(attribute_idx)
        pprint(attribute_dict)
        
        y = np.array(df['PRINC_SURG_PROC_CODE'])
        X = np.matrix(df.drop(columns='PRINC_SURG_PROC_CODE'))
        print(X.shape, y.shape)
        
        max_attr_vals = np.max(X, axis=0)
        X = X / max_attr_vals
        max_attr_vals = np.squeeze(np.array(max_attr_vals))
        print(max_attr_vals, max_attr_vals.shape)
        
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        pickle.dump([attribute_idx, attribute_dict, max_attr_vals], open(DATA_PATH+self.dataset_name+'_feature_desc.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--preprocess', type=int, default=0)
    parser.add_argument('--binarize', type=int, default=0)
    args = parser.parse_args()

    ds = PreprocessDataset(args.dataset)
    
    if args.preprocess:
        ds.preprocess()
    
    if args.binarize:
        ds.binarize()
