#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy.special import erf
import re, sqlite3, os

class IID():
    '''
    A script for auto calibrating the ion identification result based on the input Schottky spectrum
    '''
    # CODATA 2018
    c = 299792458 # speed of light in m/s
    e = 1.602176634e-19 # elementary charge in C
    me = 5.48579909065e-4 # electron mass in u
    u2kg = 1.66053906660e-27 # amount of kg per 1 u
    MeV2u = 1.07354410233e-3 # amount of u per 1 MeV

    # time unit
    time_unit = {'Yy': 31536000*1e24, 'Zy': 31536000*1e21, 'Ey': 31536000*1e18, 'Py': 31536000*1e15, 'Ty': 31536000*1e12, 'Gy': 31536000*1e9, 'My': 31536000*1e6, 'ky': 31536000*1e3, 'y': 31536000, 'd': 86400, 'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18, 'zs': 1e-21, 'ys': 1e-24}

    
    def __init__(self, lppion, cen_freq, span, win_len, gamma_t, L_CSRe, delta_Brho_over_Brho, min_sigma_t, verbose=False):
        '''
        extract all the secondary fragments and their respective yields calculated by LISE++
        (including Mass, Half-life, Yield of all the fragments)
        lppion:     LISE++ output file to be loaded
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        win_len:    window length of the spectrum
        gamma_t:    the gamma_t value for the isochronous mode
        L_CSRe:     circumference of CSRe in m, default value 128.8
        min_sigma_t:            the minimum sigma t for isochronous mode, [ps]
        delta_Brho_over_Brho:   the ΔBrho/Brho for isochronous mode, [%] (assuming that only 99.73% of ions entering the ring because of the cut of Brho limitation, delta Brho over Brho approx. 6-sigma of the ion Brho distributio)
        '''
        print('iid: initial...')
        self.cen_freq = cen_freq # MHz
        self.span = span # kHz, sampling rate = 1.25 * span
        self.win_len = win_len
        self.gamma_t = gamma_t
        self.delta_Brho_over_Brho = delta_Brho_over_Brho # %
        self.min_sigma_t = min_sigma_t # ps
        self.L_CSRe = L_CSRe # m
        self.verbose = verbose
        self.conn = sqlite3.connect("./ionic_data.db")
        self.cur = self.conn.cursor()

        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_name = [(part[0],) for part in self.cur.fetchall() if (part[0] != 'IONICDATA')]
        if len(table_name) > 0:
            self.cur.executescript(';'.join(["DROP TABLE IF EXISTS %s" %i for i in table_name]))
            self.conn.commit()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS LPPDATA
                (A          INT         NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Q           INT         NOT NULL,
                ION         TEXT        NOT NULL,
                YIELD       REAL);''')
        self.cur.execute("DELETE FROM LPPDATA")
        self.conn.commit()
        with open(lppion, encoding='latin-1') as lpp:
            while True:
                line = lpp.readline().strip()
                if line == "[D4_DipoleSettings]":
                    self.Brho = float(lpp.readline().strip().split()[2]) # Tm
                elif line == "[Calculations]":
                    break
            for line in lpp:
                segment = line.strip().split(',')[0].split()
                A, element, Q = re.split("([A-Z][a-z]?)", segment[0]+segment[-2][:-1])
                self.cur.execute("INSERT INTO LPPDATA (A,ELEMENT,Q,ION,YIELD) VALUES (?,?,?,?,?)", (A, element, Q, ''.join([A,element,Q]), segment[-1][1:]))
            self.conn.commit()
        # reset yield for both fission (including pps for AFhihg, AFmid and AFlow) and PF processing
        result = self.cur.execute("SELECT sum(YIELD), ION FROM LPPDATA GROUP BY ION").fetchall()
        self.cur.executemany("UPDATE LPPDATA SET YIELD=? WHERE ION=?", result)
        self.conn.commit()
        self.cur.executescript("""
            CREATE TABLE TEMPTABLE as SELECT DISTINCT * FROM LPPDATA;
            DROP TABLE LPPDATA;
            ALTER TABLE TEMPTABLE RENAME TO LPPDATA;""")
        self.conn.commit()
        self.calc_isochronous_peak()
        print('iid: initial complete')

    def prepare_result(self):
        self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='OBSERVEDION'")
        if self.cur.fetchone()[0] == 1:
            self.cur.execute("DROP TABLE OBSERVEDION")
            self.conn.commit()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS OBSERVEDION
                (ID         INT,
                A           INT         NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Q           INT         NOT NULL,
                Z           INT         NOT NULL,
                N           INT         NOT NULL,
                MASS        DOUBLE,
                SOURCE      TEXT,                
                ION         TEXT        NOT NULL,
                YIELD       REAL,     
                TYPE        TEXT,
                ISOMERIC    CHAR(1),
                HALFLIFE    TEXT,
                PRIMARY KEY (ID));''')
        self.cur.execute("DELETE FROM OBSERVEDION")
        self.cur.execute("INSERT INTO OBSERVEDION(A,ELEMENT,Q,Z,N,MASS,SOURCE,ION,YIELD,TYPE,ISOMERIC,HALFLIFE) \
                SELECT LPPDATA.A, LPPDATA.ELEMENT, LPPDATA.Q, IONICDATA.Z, IONICDATA.N, IONICDATA.MASS, IONICDATA.SOURCE, LPPDATA.ION, LPPDATA.YIELD, IONICDATA.TYPE, IONICDATA.ISOMERIC, IONICDATA.HALFLIFE \
                FROM IONICDATA \
                INNER JOIN LPPDATA ON IONICDATA.Q=LPPDATA.Q AND IONICDATA.ELEMENT=LPPDATA.ELEMENT AND IONICDATA.A=LPPDATA.A")
        # reset the yield of the isometric_state
        result = self.cur.execute("SELECT YIELD, ION, ISOMERIC FROM OBSERVEDION WHERE ISOMERIC!=0").fetchall()
        ## yield(isometric_state) = yield(bare) * 10**(-isometric_state)
        #re_set = [(item[0]*10**(-int(item[2])), item[1], item[2]) for item in result]
        # yield(isometric_state) = yield(bare) * 1/2
        re_set = [(item[0]*0.5, item[1], item[2]) for item in result]
        self.cur.executemany("UPDATE OBSERVEDION SET YIELD=? WHERE ION=? AND ISOMERIC=?", re_set)
        self.conn.commit()

    def calc_isochronous_peak(self):
        '''
        calculate peak locations of the Schottky signals and ToF signals from secondary fragments visible in the pre-defined frequency range (isochronous mode)
        gamma_t:                the gamma_t value for isochronous mode
        delta_Brho_over_Brho:   the ΔBrho/Brho for isochronous mode, [%]
        '''
        self.prepare_result()
        # initial table isochronousion
        self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ISOCHRONOUSION'")
        if self.cur.fetchone()[0] == 1:
            self.cur.execute("DROP TABLE ISOCHRONOUSION")
            self.conn.commit()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS ISOCHRONOUSION
                (ION         TEXT        NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Z           INT         NOT NULL,
                N           INT         NOT NULL,
                Q           INT         NOT NULL,
                ISOMERIC    CHAR(1),
                MASS        DOUBLE,
                GAMMA       REAL,
                SOURCE      TEXT,                
                YIELD       REAL,
                TOTALYIELD  REAL,
                PEAKLOC     REAL,
                PEAKSIG     REAL,
                PEAKMAX     REAL,
                HARMONIC    INT,
                REVFREQ     REAL,     
                TYPE        TEXT,
                HALFLIFE    TEXT,
                WEIGHT      REAL);''')
        self.conn.commit()

        self.cur.execute("SELECT MASS, Q, YIELD, ION, ISOMERIC FROM OBSERVEDION")
        mass, Q, ion_yield, ion, isometric_state = np.array(self.cur.fetchall()).T
        mass, Q, ion_yield = mass.astype(np.float64), Q.astype(np.float64), ion_yield.astype(np.float64)
        gamma_beta = self.Brho / mass * Q / self.c / self.u2kg * self.e
        beta = gamma_beta / np.sqrt(1 + gamma_beta**2)
        gamma = 1 / np.sqrt(1 - beta**2)
        #energy = (gamma - 1) / self.MeV2u # MeV/u
        rev_freq = beta * self.c / self.L_CSRe * 1e-6 # MHz
        weight = ion_yield * Q**2 * rev_freq**2
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        i = 0
        while (i < len(ion)):
            temp = self.cur.execute("SELECT ION,ELEMENT,N,Z,Q,ISOMERIC,MASS,SOURCE,YIELD,TYPE,HALFLIFE FROM OBSERVEDION WHERE ION=? AND ISOMERIC=?", (ion[i],isometric_state[i])).fetchone()
            # drop the ion of life time < 1 ms
            if self.life_transform(temp[-1]) <= 1 * self.time_unit['ms']:
                i += 1
                continue
            harmonics = np.arange(np.ceil(lower_freq/rev_freq[i]), np.floor(upper_freq/rev_freq[i])+1).astype(int)
            peak_sig = np.abs(1 / gamma[i]**2 - 1 / self.gamma_t**2) * self.delta_Brho_over_Brho / 6 * rev_freq[i] * 10 * harmonics + self.min_sigma_t * rev_freq[i]**2 * 1e-3 * harmonics # kHz
            # filter harmonics
            if len(harmonics) == 1 and harmonics[-1] > 0:
                self.cur.execute("INSERT INTO ISOCHRONOUSION(ION,ELEMENT,N,Z,Q,ISOMERIC,MASS,SOURCE,YIELD,TYPE,HALFLIFE,GAMMA,WEIGHT,PEAKLOC,PEAKSIG,PEAKMAX,REVFREQ,HARMONIC) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (*temp, gamma[i], weight[i], (harmonics[-1]*rev_freq[i]-self.cen_freq)*1e3, peak_sig[-1], weight[i]*erf(self.span*1.25/self.win_len/peak_sig[-1]/np.sqrt(2)/2), rev_freq[i], int(harmonics[-1])))
            elif len(harmonics) > 1:
                re_set = [(*temp, gamma[i], weight[i], (harmonics[j]*rev_freq[i]-self.cen_freq)*1e3, peak_sig[j], weight[i]*erf(self.span*1.25/self.win_len/peak_sig[j]/np.sqrt(2)/2), rev_freq[i], int(harmonics[j])) for j in range(len(harmonics)) if int(harmonics[j]) > 0]
                self.cur.executemany("INSERT INTO ISOCHRONOUSION(ION,ELEMENT,N,Z,Q,ISOMERIC,MASS,SOURCE,YIELD,TYPE,HALFLIFE,GAMMA,WEIGHT,PEAKLOC,PEAKSIG,PEAKMAX,REVFREQ,HARMONIC) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", re_set)
            else:
                pass
            i += 1
        self.conn.commit()
        # reset total yield for only .lpp result (containing bare, H-like, etc. but not containing isomers)
        result = self.cur.execute("SELECT DISTINCT N, Z FROM ISOCHRONOUSION").fetchall()
        result = [self.cur.execute("SELECT sum(yield), N, Z FROM OBSERVEDION WHERE N=? AND Z=? AND ISOMERIC='0'", _result).fetchone() for _result in result]
        self.cur.executemany("UPDATE ISOCHRONOUSION SET TOTALYIELD=? WHERE N=? AND Z=?", result)
        self.conn.commit()
                                
    def update_gamma_t(self, gamma_t):
        '''
        using the specific gamma_t to update whole data
        '''
        self.gamma_t = gamma_t
        self.calc_isochronous_peak()
        
    def update_delta_Brho_over_Brho(self, delta_Brho_over_Brho):
        '''
        using the specific ΔΒρ/Βρ to update whole data
        '''
        self.delta_Brho_over_Brho = delta_Brho_over_Brho
        self.calc_isochronous_peak()
        
    def update_min_delta_t(self, min_sigma_t):
        '''
        using the specific minimum sigma t to update whole data
        '''
        self.min_sigma_t = min_sigma_t
        self.calc_isochronous_peak()
        
    def update_cen_freq(self, cen_freq,):
        '''
        using the specific center frequency [MHz] to update whole data
        '''
        self.cen_freq = cen_freq # [MHz]
        self.calc_isochronous_peak()
        
    def update_span(self, span):
        '''
        using the specific span [kHz] to update whole data
        '''
        self.span = span # [kHz]
        self.calc_isochronous_peak()

    def update_L_CSRe(self, L_CSRe):
        '''
        using the specific length of Ring [m] to update whole data
        '''
        self.L_CSRe = L_CSRe # [m]
        self.calc_isochronous_peak()
        
    def update_win_len(self, win_len):
        '''
        using the specific length of Ring [m] to update whole data
        '''
        self.win_len = win_len 
        self.calc_isochronous_peak()

    def calibrate_Brho(self, Brho):
        '''
        using the measured Brho with the identified ion to calibrate
        Brho:       the magnetic rigidity of the target ion in Tm
        '''
        self.Brho = Brho
        self.calc_isochronous_peak()

    def calibrate_peak_loc(self, ion, isometric_state, peak_loc, harmonic):
        '''
        using the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        isometric_state: a integer of isometric state, e.g., 0
        peak_loc:   peak location in kHz after deduction of the center frequency
        harmonic:   harmonic number
        '''
        mass, Q = self.cur.execute("SELECT MASS, Q FROM OBSERVEDION WHERE ION=? AND ISOMERIC=?", (ion, isometric_state)).fetchone()
        rev_freq = (self.cen_freq + peak_loc/1e3) / harmonic # MHz
        beta = rev_freq * self.L_CSRe / self.c * 1e6
        gamma = 1 / np.sqrt(1 - beta**2)
        self.Brho = gamma * beta * mass / Q * self.c * self.u2kg / self.e # Tm
        return self.Brho

    def life_transform(self, half_life):
        '''
        transform the halflife in SQLite Table to lifetime [s]
        '''
        temp = half_life.split() # input half life in str
        if len(temp) == 1:
            if temp[0] == 'stbl':
                return 1e64
            else:
                return -1
        else:
            if temp[-1].isalpha():
                try:
                    temp_0 = float(temp[0]) / np.log(2) * self.time_unit[temp[-1]]
                except:
                    temp_0 = float(''.join(re.split(r"(\d+)", temp[0])[1:])) / np.log(2) * self.time_unit[temp[-1]]
                return temp_0
            else:
                return -1



if __name__ == "__main__":
    iid = IID("./GSI_133Sn_setting_v3.lpp", 243.5, 3000, 4096, 0.4, 1.4098)
