from sqlalchemy import create_engine
from IPython.core.debugger import Tracer
from Levenshtein import distance
from scipy.sparse import csr_matrix
import pandas as pd
import numba as nb
import numpy as np
import re
import time


class data_tools:
    def __init__(self, config) -> None:
        self.readdb = config['readdb']
        if self.readdb == 1:
            self.engine = create_engine(
                f"mysql+pymysql://{config['user']}:{config['passwd']}@{config['host']}:{config['port']}", pool_recycle=3600)
            self.dbc = self.engine.connect()
            self.dbs = config['dbs']

    def __del__(self):
        if self.readdb == 1:
            self.dbc.close()

    def aggregate_designation(self, df, designation):
        row = df.iloc[0]
        codes = []
        for _, row in df.iterrows():
            id = row.engine_designation_id
            if not np.isnan(id):
                codes.append(
                    designation.loc[row.engine_designation_id]['name'])
        return pd.Series({'id': row.id_x, 'designations': codes})

    def wash_aspiration(self, a):
        if re.match('(None|n/a|-)', a):
            return ''
        a = re.sub(r'I/C', 'intercooler', a)
        m = re.search('(turbo|superchaged)', a, re.IGNORECASE)
        if not m:
            return 'naturally aspirated'
        a = re.sub(r'I/C', 'intercooler', a)
        a = re.sub('[A-Z]+ ', '', a)
        a = re.sub('[\/-]', ' ', a)
        return self.sort_phrase(a)

    def wash_years(self, df):
        df['from'] = pd.to_datetime(df["from"], format='%Y-%m-%d').dt.year
        df['to'] = pd.to_datetime(df["to"], format='%Y-%m-%d').dt.year
        df['to'].fillna(df['from'], inplace=True)
        df['from'].fillna(df['to'], inplace=True)
        pass

    def sort_phrase(self, phrase):
        words = [word.lower() for word in phrase.split()]
        words.sort()
        return ' '.join(words)

    def get_model_codes(self, vehicle_id, df):
        pivots = df['modelcodes_fullvehicle'][df['modelcodes_fullvehicle'].full_vehicle_id == vehicle_id]
        codes = []
        for _, row in pivots.iterrows():
            codes.append(df['model_codes'].loc[row.model_code_id, 'name'])
        return codes

    def wash_model_codes(self, vehicle_df, df):
        vehicle_df['model_code'] = vehicle_df.apply(
            lambda x: self.get_model_codes(x.name, df), axis=1)
        pass

    def read_df_from_table(self):
        frm = {'engine': {}, 'supercheap': {}}
        frm['super'] = self.get_indexed_table(
            f"select id, UPPER(make) as make, LOWER(model) as model, year, series, detail, engine from {self.dbs[0]}")
        frm['vehicle'] = self.get_indexed_table(
            f"select * from {self.dbs[1]}.uvdb_full_vehicles where vehicle_type_id = 1")
        frm['engine']['defined'] = self.get_indexed_table(
            f"select * from {self.dbs[1]}.uvdb_defined_engines")
        frm['engine']['base'] = self.get_indexed_table(
            f"select * from {self.dbs[1]}.uvdb_engine_bases")
        frm['engine']['designation'] = self.get_indexed_table(
            f"select id, UPPER(name) as name from {self.dbs[1]}.uvdb_engine_designations")
        frm['engine']['defined_designation'] = self.get_indexed_table(
            f"select * from {self.dbs[1]}.uvdb_defined_engine_designations")
        frm['makes'] = self.get_indexed_table(
            f"select id, UPPER(name) as name from {self.dbs[1]}.uvdb_makes where vehicle_type_id = 1")
        frm['models'] = self.get_indexed_table(
            f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_models")
        # frm['submodels'] = self.get_indexed_table(
        #     f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_submodels")
        # frm['doors'] = self.get_indexed_table(
        #     f"select id, name from {self.dbs[1]}.uvdb_body_num_doors")
        frm['bodies'] = self.get_indexed_table(
            f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_body_types")
        frm['drives'] = self.get_indexed_table(
            f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_drive_types")
        # frm['regions'] = self.get_indexed_table(
        #     f"select id, UPPER(name) as name from {self.dbs[1]}.uvdb_regions")
        # frm['transmissions'] = self.get_indexed_table(
        #     f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_transmission_control_types")
        frm['valves'] = self.get_indexed_table(
            f"select id, name from {self.dbs[1]}.uvdb_valves")
        frm['heads'] = self.get_indexed_table(
            f"select id, UPPER(name) as name from {self.dbs[1]}.uvdb_cylinder_head_types")
        frm['aspirations'] = self.get_indexed_table(
            f"select id, LOWER(name) as name from {self.dbs[1]}.uvdb_aspirations")
        frm['delivery'] = self.get_indexed_table(
            f"select id, UPPER(name) as name from {self.dbs[1]}.uvdb_fuel_delivery_subtypes")
        frm['powers'] = self.get_indexed_table(
            f"select id, kw as name from {self.dbs[1]}.uvdb_engine_power_outputs")
        frm['model_codes'] = self.get_indexed_table(
            f"select id, UPPER(name) as name, make_id from {self.dbs[1]}.uvdb_model_codes")
        frm['modelcodes_fullvehicle'] = self.get_indexed_table(
            f"select * from {self.dbs[1]}.uvdb_model_code_full_vehicle")
        frm['supercheap']['detail'] = self.generate_detail_data(frm['super'])
        frm['supercheap']['engine'] = self.generate_engine_data(frm['super'])
        return frm

    def get_indexed_table(self, sql, keep_index_column=True):
        df = pd.read_sql(sql, self.dbc).set_index('id')
        if keep_index_column:
            df['id'] = df.index
        return df

    def generate_engine_data(self, super_df, keep_index_column=True):
        df = super_df.apply(lambda x: self.get_engine(
            x.engine, x.name), axis=1).set_index('id')
        if keep_index_column:
            df['id'] = df.index
        return df

    def get_engine(self, engine, id):
        debug = 0
        original = engine
        engine = re.sub('[.\(\)]', '', engine)
        engine = re.sub('2 STROKE', '', engine, re.IGNORECASE)

        power = ''
        m = re.search('\{(\d+)KW\}', engine, re.IGNORECASE)
        if m:
            power = m[1]
            engine = f"{m.string[:m.start()]} {m.string[m.end():]}"
        elif debug == 1:
            print('pw: '+original)

        capacity = ''
        m = re.search('^(\d+)cc,', engine)
        if m:
            capacity = m[1]
            engine = f"{m.string[:m.start()]} {m.string[m.end():]}".strip()
        elif debug == 1:
            print('cc: '+original)

        head = ''
        m = re.search(
            '(?: |^)(SOHC|DOHC|OHV|CIH|N/R|N/A|U/K|L-HEAD|F-HEAD)(?: |$)', engine)
        if m:
            head = m[1]
            engine = f"{m.string[:m.start()]} {m.string[m.end():]}"
        elif debug == 1:
            print('head: '+original)
        engine = re.sub('[-]', '', engine)

        delivery = ''
        m = re.search('(?: |^)(?:(\w+) )?(\w+)$', engine)
        if m:
            if m[2] == 'Inj' or (m[1] is not None and re.match('^(TWIN|TRIPLE|4B|ELEC)$', m[1])):
                delivery = f"{m[1]} {m[2]}"
                engine = f"{m.string[:m.start()]} {m.string[m.end():]}"
            else:
                delivery = m[2]
                engine = f"{m.string[:m.start()]} {m[1]} {m.string[m.end():]}".strip(
                )
        if debug == 1 and delivery == '':
            print('delivery: '+original)

        valves = ''
        engine = engine[::-1]
        m = re.search('(?: |^)[vV](\d+)(?: |$)', engine)
        if m:
            valves = m[1][::-1]
            engine = f"{m.string[:m.start()]} {m.string[m.end():]}".strip()[
                ::-1]
        elif debug == 1:
            print('valves: '+original)
            engine = engine[::-1]
        else:

            engine = engine[::-1]

        designation = ''
        block = ''
        cylinders = ''
        engine = engine[::-1]
        m = re.search(
            '(?: |^)(\d{1,2})([a-zA-Z]|YRATOR)(?: |$)', engine, re.IGNORECASE)
        if m:
            designation = m.string[m.end():].strip()[::-1]
            cylinders = m[1][::-1]
            block = m[2][::-1]
            engine = f"{m.string[:m.start()]}".strip()[
                ::-1]
        elif debug == 1:
            print('designation, block and cylinders: '+original)
            engine = engine[::-1]
        else:
            engine = engine[::-1]

        aspiration = engine
        if debug == 1 and aspiration == '':
            print('aspiration: '+original)
        aspiration = 'naturally aspirated' if aspiration == '' else aspiration
        return pd.Series({'id': id, 'capacity': capacity, 'designation': designation, 'block': block, 'cylinders': cylinders, 'valves': valves, 'head': head, 'aspiration': aspiration, 'delivery': delivery, 'power': power})

    def generate_detail_data(self, super_df, keep_index_column=True):
        df = super_df.apply(lambda x: self.get_detail(
            x.detail, x.series, x.name), axis=1).set_index('id')
        if keep_index_column:
            df['id'] = df.index
        return df

    def get_detail(self, detail, series, id):
        debug = 0
        drive = ''
        m = re.search('[ ,]([4AFR]WD)(?:[ ,]|$)', detail)
        if m:
            drive = m[1]
        elif debug == 1:
            print('drive: '+detail)
        transmission = ''
        m = re.search(', +([^ ]+)$', detail)
        if m:
            transmission = m[1]
            m1 = re.search('^[4AFR]WD', transmission)
            if m1:
                transmission = ''
        if debug == 1 and transmission == '':
            print('trans: '+detail)
        region = ''
        m = re.search('\[([^\]]+)\]', detail)
        if m:
            region = m[1]
        elif debug == 1:
            print('region: '+detail)

        doors = ''
        m = re.search('(?:^|[ ,])(\d)D[ ,]', detail)
        if m:
            doors = m[1]
        elif debug == 1:
            print('doors: '+detail)

        body = ''
        m = re.search('(?:^|, +)(?:\dD )([^,]+)', detail)
        if m:
            body = m[1].strip()
        elif debug == 1:
            print('body: '+detail)

        model_code = ''
        m = re.search('^(.{' + str(len(series)//2) + '})', series)
        if m:
            model_code = m[1].strip()
        # model_sub = model
        # m = re.search('^(?:([ .a-zA-Z0-9-]+), +([ .a-zA-Z0-9-]+),)', detail)
        # if m:
        #     s1 = m[1]
        #     s2 = m[2]
        #     m1 = re.search('^\dD [ ,]', s1)
        #     if m1:
        #         s1 = ''
        #     m2 = re.search('^\dD[ ,]', s2)
        #     if m2:
        #         s2 = ''
        #     if s1 != '' or s2 != '':
        #         model_sub += f" {s1} {s2}".strip()
        #         model_sub = self.sort_phrase(model_sub)
        # if model_sub == model and debug == 1:
        #     print('model_sub: '+detail)

        return pd.Series({'id': id, 'model_code': model_code, 'doors': doors, 'body': body, 'drive': drive, 'region': region, 'transmission': transmission})

    def get_vehicle_feature_list(self, vehicle_features):
        defined_vehicle_features = []
        for field in vehicle_features.values():
            if '.' in field:
                continue
            else:
                defined_vehicle_features.append(field)
        return defined_vehicle_features

    def get_engine_feature_list(self, engine_features):
        defined_engine_features = ['base_id', 'designations']
        for field in engine_features.values():
            if '.' in field:
                continue
            else:
                defined_engine_features.append(field)
        return defined_engine_features

    def transform_data_from_tables(self, df, full_vehicle_features, supercheap_vehicle_features, engine_features):
        data = {}
        # 3.1 Extracting and Merging
        # Extracting
        data['full_vehicle'] = df['vehicle'][self.get_vehicle_feature_list(
            full_vehicle_features)]
        # Merging make
        data['full_vehicle'] = data['full_vehicle'].merge(
            df['makes'], how='left', left_on='make_id', right_index=True)
        data['full_vehicle'].rename(
            {'name': 'make', 'drive_type_id': 'drive_id'}, axis=1, inplace=True)
        data['full_vehicle'].drop(['id', 'make_id'], axis=1, inplace=True)
        # Merging model
        data['full_vehicle'] = data['full_vehicle'].merge(
            df['models'], how='left', left_on='model_id', right_index=True)
        data['full_vehicle'].rename({'name': 'model'}, axis=1, inplace=True)
        data['full_vehicle'].drop(['id', 'model_id'], axis=1, inplace=True)
        pd.options.mode.chained_assignment = None  # default='warn'
        # Washing years
        self.wash_years(data['full_vehicle'])
        # Getting model codes
        self.wash_model_codes(data['full_vehicle'], df)
        pd.options.mode.chained_assignment = 'warn'
        # Merging body
        data['full_vehicle'] = data['full_vehicle'].merge(
            df['bodies'], how='left', left_on='body_type_id', right_index=True)
        data['full_vehicle'].rename({'name': 'body'}, axis=1, inplace=True)
        data['full_vehicle'].drop(['body_type_id', 'id'], axis=1, inplace=True)
        data['full_vehicle'].fillna(-1, inplace=True)
        data['full_vehicle'] = data['full_vehicle'].reindex(
            sorted(data['full_vehicle'].columns), axis=1).sort_index()

        # 4. Defined Engine
        tmp = df['engine']['defined'].merge(
            df['engine']['defined_designation'], how='left', left_index=True, right_on='defined_engine_id')
        data['defined_engine'] = tmp.reset_index().groupby('id_x').apply(
            lambda x: self.aggregate_designation(x, df['engine']['designation']))
        extended_defined_engine = data['defined_engine'].merge(
            df['engine']['defined'], left_index=True, right_index=True)
        extended_defined_engine = extended_defined_engine.merge(
            df['engine']['base'], left_on='base_id', right_index=True)
        pd.options.mode.chained_assignment = None  # default='warn'

        data['defined_engine'] = extended_defined_engine[self.get_engine_feature_list(
            engine_features)]
        data['defined_engine']['id'] = data['defined_engine'].index
        data['defined_engine'].index.name = 'id'
        data['defined_engine'].drop(['base_id'], axis=1, inplace=True)
        data['defined_engine'].rename(
            {'cc': 'capacity', 'block_type': 'block', 'cylinder_head_type_id': 'head_id'}, axis=1, inplace=True)
        # merge valves
        data['defined_engine'] = data['defined_engine'].merge(
            df['valves'], how='left', left_on='valves_id', right_index=True)
        data['defined_engine'].rename(
            {'id_x': 'id', 'name': 'valves'}, axis=1, inplace=True)
        data['defined_engine'].drop(
            ['id_y', 'valves_id'], axis=1, inplace=True)
        # merge aspiration
        data['defined_engine'] = data['defined_engine'].merge(
            df['aspirations'], how='left', left_on='aspiration_id', right_index=True)
        data['defined_engine'].rename(
            {'id_x': 'id', 'name': 'aspiration'}, axis=1, inplace=True)
        data['defined_engine'].drop(
            ['id_y', 'aspiration_id'], axis=1, inplace=True)
        # merge delivery
        data['defined_engine'] = data['defined_engine'].merge(
            df['delivery'], how='left', left_on='fuel_delivery_subtype_id', right_index=True)
        data['defined_engine'].rename(
            {'id_x': 'id', 'name': 'delivery'}, axis=1, inplace=True)
        data['defined_engine'].drop(
            ['id_y', 'fuel_delivery_subtype_id'], axis=1, inplace=True)
        # merge output
        df['powers']['name'].fillna(value=np.nan, inplace=True)
        data['defined_engine']['power_output_id'].fillna(-1, inplace=True)
        data['defined_engine'] = data['defined_engine'].merge(
            df['powers'], how='left', left_on='power_output_id', right_index=True)
        data['defined_engine'].rename(
            {'id_x': 'id', 'name': 'power'}, axis=1, inplace=True)
        data['defined_engine'].drop(
            ['id_y', 'power_output_id', 'id'], axis=1, inplace=True)

        data['defined_engine'].sort_index(inplace=True)
        data['defined_engine'] = data['defined_engine'].reindex(
            sorted(data['defined_engine'].columns), axis=1).sort_index()
        pd.options.mode.chained_assignment = 'warn'

        # 5. Supercheap data
        # 5.1 Transforming tables
        # 5.1.1 makes
        df['makes']['name'].replace('[- _]', '', regex=True, inplace=True)
        df['super']['make'].replace('[- _]', '', regex=True, inplace=True)
        name_mapping = {'ASIA': 'ASIAMOTORS', 'BMWALPINA': 'ALPINA', 'CITROEN': 'CITROËN', 'MERCEDESAMG': 'MERCEDESBENZ',
                        'MERCEDESMAYBACH': 'MERCEDESBENZ', 'PRAIRIE': 'NISSAN', 'PRINCE': 'NISSAN', "RANGEROVER": "LANDROVER",
                        'STATESMAN': 'HOLDEN', 'TRD': 'TOYOTA', 'ZXAUTO': 'ZHONGXING(AUTO)'}
        df['makes']['name'].replace(name_mapping, inplace=True)
        df['super']['make'].replace(name_mapping, inplace=True)

        # 5.2 Merging tables to super cheap data
        # 5.2.1 Makes
        extended_super = df['super'].merge(
            df['makes'], how='left', left_on='make', right_on='name')
        extended_super.rename(
            {'id_x': 'id', 'id_y': 'make_id'}, axis=1, inplace=True)
        extended_super.drop('name', axis=1, inplace=True)

        # 5.3 Building Extended Detail table and Engine table
        # 5.3.1 Merging tables to Detail
        # extended_detail = df['supercheap']['detail'].merge(df['doors'], how='left', left_on='doors', right_on='name')
        # extended_detail.rename({'id_x': 'id', 'id_y': 'doors_id'}, axis=1, inplace=True)
        # extended_detail.drop('name', axis=1, inplace=True)
        # extended_detail['body'] = extended_detail.body.str.lower()
        extended_detail = df['supercheap']['detail']
        # 5.3.1.1 Transforming bodies
        name_mapping = {
            'suv': 'sport utility',
            'utili': 'crew cab pickup',
            'cab c': 'cab & chassis',
            'hardt': 'hardtop',
            'wells': 'crew cab pickup',
            'ambul': 'u/k',
            'fastb': 'fastback',
            'liftb': 'hatchback',
            'troop': 'wagon',
            'resin': 'sport utility',
            'soft': 'sport utility',
            'dump': 'u/k',
            'roads': 'convertible',
            'tray': 'crew cab pickup',
            'sweep': 'step van - unknown',
            'minib': 'mini passenger van',
            'campe': 'van camper',
            'conve': 'convertible',
            'hatch': 'hatchback',
            'roadster': 'convertible',
            'custo': 'sedan',
            'limou': 'limousine',
            'hears': 'hearse',
        }
        extended_detail.body = extended_detail.body.str.strip()
        extended_detail.body.replace(name_mapping, inplace=True)
        extended_detail = pd.merge(
            extended_detail, df['bodies'], how='left', left_on='body', right_on='name')
        extended_detail.rename(
            {'id_x': 'id', 'id_y': 'body_id'}, axis=1, inplace=True)
        extended_detail.drop('name', axis=1, inplace=True)

        name_mapping = {
            'FWD': 'front-wheel drive',
            'RWD': 'rear-wheel drive',
            '4WD': 'four-wheel drive',
            'AWD': 'all-wheel drive'
        }
        extended_detail.drive.replace(name_mapping, inplace=True)
        extended_detail = pd.merge(
            extended_detail, df['drives'], how='left', left_on='drive', right_on='name')
        extended_detail.rename(
            {'id_x': 'id', 'id_y': 'drive_id'}, axis=1, inplace=True)
        extended_detail.drop('name', axis=1, inplace=True)
        name_mapping = {
            'USA': 'UNITED STATES',
            'UK': 'UNITED KINGDOM',
            'GREAT BRITAIN': 'UNITED KINGDOM',
        }
        name_mapping = {
            'MT': 'manual',
            'AT': 'automatic',
            'CVT': 'automatic cvt'
        }
        # extended_detail.transmission.replace(name_mapping, inplace=True)
        # extended_detail = pd.merge(extended_detail, df['transmissions'], how='left', left_on='transmission', right_on='name')
        # extended_detail.rename({'id_x': 'id', 'id_y': 'transmission_id'}, axis=1, inplace=True)
        # extended_detail.drop('name', axis=1, inplace=True)
        # extended_detail[extended_detail['transmission_id'].isna()]["transmission"].unique()
        extended_detail = extended_detail.set_index('id')
        extended_detail = extended_detail.merge(
            df['super'], left_index=True, right_index=True)
        data['super_vehicle'] = extended_detail[supercheap_vehicle_features]
        data['super_vehicle'].body = data['super_vehicle'].body.str.lower()
        data['super_vehicle'] = data['super_vehicle'].reindex(
            sorted(data['super_vehicle'].columns), axis=1).sort_index()

        # 5.4 Merging tables to Engine
        # 5.4.1 Merging head
        name_mapping = {
            'I': 'L',
            'F': 'H',
            'ROTARY': 'R'
        }
        df['supercheap']['engine'].block.replace(name_mapping, inplace=True)
        df['aspirations'].name = df['aspirations'].name.apply(
            self.wash_aspiration)
        name_mapping = {
            'Direct Inj': 'DI',
            'MPFI': 'MFI',
            'SPFI': 'SFI',
            'CARB': '1BBL',
            'EFI': 'DI',
            'CRD': 'COMMON RAIL',
            'Diesel Inj': 'DI',
            '4CARB': '4BBL',
            'TWIN CARB': '2BBL',
            '3CARB': '3BBL',
            '4B CARB': '4BBL',
            'TRIPLE CARB': '3BBL',
            '6CARB': '6BBL',
            'GDI': 'DI',
            'EDI': 'DI',
            'ELEC CARB': '1BBL'
        }
        data['super_engine'] = df['supercheap']['engine'].merge(
            df['heads'], how='left', left_on='head', right_on='name')
        data['super_engine']['aspiration'] = data['super_engine']['aspiration'].apply(
            self.wash_aspiration)
        data['super_engine'].rename(
            {'id_x': 'id', 'id_y': 'head_id'}, axis=1, inplace=True)
        data['super_engine'] = data['super_engine'].set_index('id')
        data['super_engine'].drop(['name', 'head'], axis=1, inplace=True)
        data['super_engine'].delivery.replace(name_mapping, inplace=True)
        data['super_engine'] = data['super_engine'].reindex(
            sorted(data['super_engine'].columns), axis=1).sort_index()
        return data

    def convert_column_datatype(self, col, dtype, df):
        if dtype == 'string':
            empty = ''
        else:
            empty = -1
        df[col].replace('', empty, inplace=True)
        df[col].replace('N/A', empty, inplace=True)
        df[col] = df[col].fillna(empty)
        df[col] = df[col].astype(dtype)

    def belong_compare(self, a, arr, **kw):
        contained_score = 0.5
        if 'c_score' in kw:
            contained_score = kw['c_score']
        else:
            contained_score = -1
        if a in arr:
            return 1
        for b in arr:
            if contained_score != -1 and self.contain_compare(a, b, e_score=0, c_score=contained_score, how='right') > 0:
                return contained_score
        return 0

    def engine_belong_compare(self, a, arr_str, **kw):
        makeup = 0.3
        if 'first_makeup' in kw:
            makeup = kw['first_makeup']
        m = re.search(f'^{a}: (\d+\.\d+)', a)
        if m:
            return makeup + m[1] / 100
        m = re.search(f'{a}: (\d+\.\d+)', a)
        if m:
            return m[1] / 100
        return 0

    def ratio_compare(self, a, b, **kw):
        d = max(a, b)
        if d == 0 or a == -1:
            return 1
        res = min(a, b)/d
        if res > 0.99:
            return res
        return 0

    def equal_compare(self, a, b, **kw):
        if a == b:
            return 1
        return 0

    def contain_compare(self, a, b, **kw):
        how = 'mutual'
        if 'how' in kw:
            how = kw['how']
        score = 0.5
        empty_score = 0.8
        if 'e_score' in kw:
            empty_score = kw['e_score']
        if 'c_score' in kw:
            score = kw['c_score']
        if a == '' or b == '':
            return empty_score
        if how == 'mutual':
            if a in b or b in a:
                return score
        elif how == 'left':
            if b in a:
                return score
        elif how == 'right':
            if a in b:
                return score
        return 0

    def edit_compare(self, a, b, **kw):
        d = max(len(a), len(b))
        return 1 - (distance(a, b)/d)

    def compare(self, alist, blist, compare_func, **kw):
        data = [0]
        row = [0]
        col = [0]
        time0 = time.time()
        for i, a in enumerate(alist):
            if i % 1000 == 999:
                time1 = time.time()
                print(f"\t1000 : {time1 - time0}")
            for j, b in enumerate(blist):
                if a == b:
                    value = 1
                else:
                    value = compare_func(a, b, **kw)
                if value >= 0.5:
                    data.append(value)
                    row.append(i)
                    col.append(j)
        return csr_matrix((data, (row, col)), shape=(len(alist), len(blist)))

    def calculate_engine_scores(self, data, size, engine_feature_types):
        t0 = time.time()
        # Regulating data type
        for col, dtype in sorted(engine_feature_types.items()):
            self.convert_column_datatype(col, dtype, data['super_engine'])
            self.convert_column_datatype(col, dtype, data['defined_engine'])
        scores = {}
        scores['capacity'] = self.compare(data['super_engine'].capacity.to_numpy(
        )[0: size], data['defined_engine'].capacity.to_numpy(), self.ratio_compare)
        scores['designation'] = self.compare(data['super_engine'].designation.to_numpy(
        )[0: size], data['defined_engine'].designations.to_numpy(), self.belong_compare, c_score=0.8)
        scores['block'] = self.compare(data['super_engine'].block.to_numpy(
        )[0: size], data['defined_engine'].block.to_numpy(), self.equal_compare)
        scores['cylinders'] = self.compare(data['super_engine'].cylinders.to_numpy(
        )[0: size], data['defined_engine'].cylinders.to_numpy(), self.equal_compare)
        scores['valves'] = self.compare(data['super_engine'].valves.to_numpy(
        )[0: size], data['defined_engine'].valves.to_numpy(), self.equal_compare)
        scores['head'] = self.compare(data['super_engine'].head_id.to_numpy(
        )[0: size], data['defined_engine'].head_id.to_numpy(), self.equal_compare)
        scores['aspiration'] = self.compare(data['super_engine'].aspiration.to_numpy(
        )[0: size], data['defined_engine'].aspiration.to_numpy(), self.edit_compare)
        scores['delivery'] = self.compare(data['super_engine'].delivery.to_numpy(
        )[0: size], data['defined_engine'].delivery.to_numpy(), self.contain_compare, c_score=0.6)
        scores['power'] = self.compare(data['super_engine'].power.to_numpy(
        )[0: size], data['defined_engine'].power.to_numpy(), self.ratio_compare)
        t1 = time.time()
        total = t1-t0
        print(total)
        return scores

    def get_top_k(self, m, k, left_df, right_df):
        rows = m.shape[0]
        res = np.empty([rows, 2], dtype=object)
        for i in range(rows):
            row = m.getrow(i).toarray()[0].ravel()
            top_indicies = row.argsort()[-k:]
            top_values = row[row.argsort()[-k:]]
            id_left = left_df.index[i]
            res[i][0] = id_left
            topk = ''
            for cj in range(k):
                j = k - cj - 1
                id_right = right_df.index[top_indicies[j]]
                topk += "\t"+"{0}: {1:.2f}".format(id_right, top_values[j])
            topk = re.sub('^\t', '', topk)
            res[i][1] = topk
        return res

    def compare_range(self, alist, blist, clist):
        data = [0]
        row = [0]
        col = [0]
        time0 = time.time()
        for i, a in enumerate(alist):
            if i % 10000 == 9999:
                time1 = time.time()
                print(f"{i} : {time1 - time0}")
            for j, b in enumerate(blist):
                if a >= b and a <= clist[j]:
                    value = 1
                else:
                    value = 0
                data.append(value)
                row.append(i)
                col.append(j)
        try:
            return csr_matrix((data, (row, col)), shape=(len(alist), len(blist)))
        except:
            Tracer()()

    def calculate_vehicle_scores(self,  data, size, vehicle_feature_types):
        t0 = time.time()
        # Regulating data type for vehicle data
        for col, dtype in sorted(vehicle_feature_types.items()):
            if col in data['super_vehicle']:
                self.convert_column_datatype(col, dtype, data['super_vehicle'])
            if col in data['full_vehicle']:
                self.convert_column_datatype(col, dtype, data['full_vehicle'])
        scores = {}
        scores['body'] = self.compare(data['super_vehicle'].body.to_numpy()[
                                      0: size], data['full_vehicle'].body.to_numpy(), self.contain_compare, c_score=0.6)
        scores['drive'] = self.compare(data['super_vehicle'].drive_id.to_numpy(
        )[0: size], data['full_vehicle'].drive_id.to_numpy(), self.equal_compare)
        scores['make'] = self.compare(data['super_vehicle'].make.to_numpy()[
                                      0: size], data['full_vehicle'].make.to_numpy(), self.contain_compare, c_score=0.6)
        scores['model'] = self.compare(data['super_vehicle'].model.to_numpy()[
                                       0: size], data['full_vehicle'].model.to_numpy(), self.contain_compare, c_score=0.6)
        scores['model_code'] = self.compare(data['super_vehicle'].model_code.to_numpy(
        )[0: size], data['full_vehicle'].model_code.to_numpy(), self.edit_compare)
        scores['year'] = self.compare_range(data['super_vehicle'].year[0: size],
                                            data['full_vehicle']['from'].to_numpy(), data['full_vehicle']['to'].to_numpy())
        scores['engines'] = self.compare(data['super_vehicle'].engines.to_numpy(
        )[0: size], data['full_vehicle'].defined_engine_id.to_numpy(), self.engine_belong_compare, first_makeup=0.4)
        t1 = time.time()
        total = t1-t0
        print(total)
        return scores

    def add_engine_info(self, top_engine_scores, df_super_vehicle):
        # Adding engine infomation to supercheap vehicle
        df_engine_mapping = pd.DataFrame(
            data=top_engine_scores, columns=['id', 'engines'])
        df_engine_mapping = df_engine_mapping.set_index('id')
        # Merge engine mapping to supercheap vehicles
        df_super_vehicle = df_super_vehicle.merge(
            df_engine_mapping, how='left', left_index=True, right_index=True)
        df_super_vehicle.engines.fillna('', inplace=True)
        return df_super_vehicle
