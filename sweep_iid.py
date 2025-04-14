#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr, Global

from iid_isochronous_Schottky import IID
import time, os

class SWEEP_IID(IID):
    
    def __init__(self, peak_file, lppion, cen_freq, span, win_len, gamma_t, L_CSRe=128.8):
        '''
        peak_file:      .csv file extracted from spectrum including 'exp_freq'(peaks' frequencies), 'peak_width', 'peak_height'
        '''
        self.peak_file = pd.read_csv(peak_file)
        self.peak_file['exp_freq'] = self.peak_file['exp_freq'] - cen_freq * 1e3 # kHz
        super().__init__(lppion, cen_freq, span, win_len, gamma_t, L_CSRe, delta_Brho_over_Brho=0.2, min_sigma_t=0.5, verbose=False)
        self.iid_file = 'LISE_{:}_PeakFile_{:}'.format(os.path.basename(lppion)[:-4], os.path.basename(peak_file)[:-4])
        peak_max_ind = self.peak_file['peak_height'].idxmax()
        sweep_freq_limit = self.cur.execute("SELECT REVFREQ FROM ISOCHRONOUSION WHERE YIELD = (SELECT max(YIELD) FROM ISOCHRONOUSION)").fetchone()[0] * 1e3
        sweep_spec_peaks = self.peak_file[np.abs(self.peak_file['exp_freq'] - self.peak_file['exp_freq'][peak_max_ind])<=sweep_freq_limit].copy()
        sweep_spec_peaks['sort_key'] = np.abs(sweep_spec_peaks['exp_freq']-self.peak_file['exp_freq'][peak_max_ind])
        sweep_spec_peaks.sort_values(['sort_key'], ascending=True, inplace=True, ignore_index=True)
        self.sweep_spec_peaks = sweep_spec_peaks

    def calibrate_fixedC(self, L_CSRe=128.8, init_Brho=None, delta_Brho=0.01, in_global=False, save_PID_dataframe=True):
        '''
        the method to get the best Brho for the PID result from a fixed testing orbit length
        L_CSRe:         the inital testing orbit length, [m], default 128.8
        init_Brho:      the inital testing Brho, [Tm], default None using the same Brho from .lpp setting
        delta_Brho:     the Brho loop range, |(testing Brho - init Brho)/init Brho| <= delta_Brho
        in_global:      if this method be inserted in calibrate_global, then True, default False
        '''
        t0 = time.time()
        df_PID = pd.DataFrame(columns=['peakloc', 'ID', 'harmonic', 'Brho_test', 'Brho_est', 'C_test', 'C_est', 'reduced_chisqr', 'addition_chisqr', 'skip_ionNum', 'corr_ionNum'])
        
        print('sweep peaks number in spec: {:}'.format(len(self.sweep_spec_peaks)))
        if init_Brho != None:
            self.calibrate_Brho(init_Brho)

        init_result = self.cur.execute("SELECT ION, ISOMERIC, HARMONIC, PEAKLOC FROM ISOCHRONOUSION WHERE YIELD >= 1").fetchall()
        init_Brho, init_L = self.Brho, self.L_CSRe
        if not in_global:
            # call Wolfram Kernel
            sess = WolframLanguageSession()
        else:
            sess = self.sess

        pass_status = False # control the loop for the iid
        df_i = 0 # PID dataframe writing line number
        for i_spec, row_spec in self.sweep_spec_peaks.iterrows():
            if pass_status:
                break
            ion_data = pd.DataFrame({'ion': [_temp[0] for _temp in init_result],
                'isomeric': [_temp[1] for _temp in init_result],
                'harmonic': [_temp[2] for _temp in init_result], 
                'peakloc': [_temp[3] for _temp in init_result],
                'sort_key': [np.abs(_temp[3] - row_spec['exp_freq']) for _temp in init_result]})
            ion_data.sort_values(['sort_key'], ascending=True, inplace=True, ignore_index=True)
            for i_ion, row_ion in ion_data.iterrows():
                _Brho = self.calibrate_peak_loc(row_ion['ion'], row_ion['isomeric'], row_spec['exp_freq'], row_ion['harmonic'])
                if np.abs(_Brho - init_Brho)/init_Brho > delta_Brho:
                    continue
                self.calibrate_Brho(_Brho)
                temp = self.cur.execute("SELECT ION, ISOMERIC, HARMONIC, PEAKLOC, MASS, Q FROM ISOCHRONOUSION WHERE YIELD >= 1").fetchall()
                _ion_data = pd.DataFrame({'ion': [_temp[0] for _temp in temp],
                    'isomeric': [_temp[1] for _temp in temp],
                    'harmonic': [_temp[2] for _temp in temp], 
                    'peakloc': [_temp[3] for _temp in temp],
                    'mass': [_temp[4] for _temp in temp],
                    'Q': [_temp[5] for _temp in temp]})
                chi_sqr, skip_ionNum, chi_num = 0, 0, 0
                # solve equation to get minimum of chiSqr
                expr_temp_00 = '' # inital expression
                expr_temp_01 = '' 
                expr_temp_02 = '' 
                expr_temp_03 = '' 
                expr_temp_04 = ''
                expr_temp_10 = ''
                expr_temp_11 = ''
                expr_temp_12 = ''
                for ii_spec, irow_spec in self.sweep_spec_peaks.iterrows():
                    _ind = np.abs(_ion_data['peakloc']-irow_spec['exp_freq']).idxmin()
                    if np.abs(_ion_data['peakloc'][_ind] - irow_spec['exp_freq']) <= 1.5/2/np.sqrt(2*np.log(2))*irow_spec['peak_width']: 
                        mass, Q, exp_freq = _ion_data['mass'][_ind], _ion_data['Q'][_ind], (irow_spec['exp_freq']+self.cen_freq*1e3)/_ion_data['harmonic'][_ind]*1e3
                        expr_temp_00 += ''' + {:.16f}^2 * {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2)^(3/2) '''.format(mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c)
                        expr_temp_01 += ''' + {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2 )^(1/2) '''.format(exp_freq, mass/Q/self.e*self.u2kg, self.c)
                        expr_temp_02 += ''' + ( {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:}^2 ) )^2 '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c)
                        expr_temp_03 += ''' + 1 / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2 ) '''.format(mass/Q/self.e*self.u2kg, self.c)
                        expr_temp_04 += ''' + ({:} - 1 / ( {:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(1/2) / CC)^2 '''.format(exp_freq, mass/Q/self.e*self.u2kg, self.c)
                        expr_temp_10 += ''' + 6 / CC^4 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2) - 4 * {:.16f} / CC^3 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(1/2) '''.format(mass/Q/self.e*self.u2kg, self.c, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_CC = A
                        expr_temp_11 += ''' - 4 * {:.16f}^2 / Brho^3 / CC^3 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^2 + 2 * {:.16f}^2 / Brho^3 / CC^2 * {:.16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(3/2) '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_CBrho = B
                        expr_temp_12 += ''' - 4 * {:.16f}^4 / Brho^6 / CC^2 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^3 + 6 * {:.16f}^4 / Brho^6 / CC * {:.16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(5/2) + 6 * {:.16f}^2 / Brho^4 / CC * {:16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(3/2) - 6 * {:.16f}^2 / Brho^4 / CC^2 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^2 '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_BrhoBrho = C
                        chi_num += 1
                    else:
                        skip_ionNum += 1
                        continue
                expr_temp = '''result = FindRoot[({:}) / ({:}) - ({:}) / ({:}) == 0, {{x, {:}}}];
                                Brho = x /. result;
                                CC = ({:}) / ({:}) /. result;
                                ChiSqr = ({:}) / {:};
                                judgeA = {:} > 0;
                                judgeABC = ({:}) * ({:}) - ({:})^2 > 0;'''.format(expr_temp_00[2:], expr_temp_01[2:], expr_temp_02[2:], expr_temp_03[2:], _Brho, expr_temp_03[2:], expr_temp_01[2:], expr_temp_04[2:], chi_num, expr_temp_10[2:], expr_temp_10[2:], expr_temp_12, expr_temp_11)
                if chi_num > 2:
                    sess.evaluate(wlexpr(expr_temp))
                    iid_Brho = sess.evaluate(Global.Brho)
                    iid_L = sess.evaluate(Global.CC)
                    iid_ChiSqr = sess.evaluate(Global.ChiSqr)
                    iid_judgeA = sess.evaluate(Global.judgeA)
                    iid_judgeABC = sess.evaluate(Global.judgeABC)
                    print('peakloc: {:}, id: {:}({:}), harmonic: {:}, Brho: {:.7f} Tm, C: {:.4f}, skip num: {:}, included num: {:}\nChiSqr: {:.5f}, A>0: {:}, AC-B^2>0: {:}, Reduced_ChiSqr: {:}, Additional_ReChiSqr: {:}'.format(row_spec['exp_freq'], row_ion['ion'], row_ion['isomeric'],  row_ion['harmonic'], iid_Brho, iid_L, skip_ionNum, chi_num, iid_ChiSqr, iid_judgeA, iid_judgeABC, iid_ChiSqr/(chi_num-1), iid_ChiSqr/(chi_num-1)*skip_ionNum**2/chi_num**2))
                    item_list = [row_spec['exp_freq'], "{:}({:})".format(row_ion['ion'], row_ion['isomeric']), row_ion['harmonic'], _Brho, iid_Brho, init_L, iid_L, iid_ChiSqr/(chi_num-1), iid_ChiSqr/(chi_num-1)*skip_ionNum**2/chi_num**2, skip_ionNum, chi_num]
                    df_PID.loc[df_i] = item_list
                    df_i += 1
                else:
                    pass
            if pass_status:
                pass
            else:
                chisqr_idxmin = df_PID[df_PID['peakloc']==row_spec['exp_freq']]['addition_chisqr'].idxmin()
                if (chisqr_idxmin == df_PID[df_PID['peakloc']==row_spec['exp_freq']]['skip_ionNum'].idxmin()) or (df_PID['skip_ionNum'][chisqr_idxmin] < np.average(df_PID[df_PID['peakloc']==row_spec['exp_freq']]['skip_ionNum'].values) - np.std(df_PID[df_PID['peakloc']==row_spec['exp_freq']]['skip_ionNum'].values)):
                    pass_status = True

        if pass_status == False:
            print("no suitable iid result. Please check .lpp files.")
            print('time for calibrating with fixed orbit length: {:.3f} sec'.format(time.time()-t0))
            if not in_global:
                sess.stop()
            return 

        if save_PID_dataframe:
            df_PID.to_csv('./PID_{:}_bestBrhoC_{:}.csv'.format(init_L,self.iid_file), encoding='utf-8', index=True)
            print('iid results is saved in ./PID_{:}_bestBrhoC_{:}.csv'.format(init_L,self.iid_file))

        print('time for calibrating with fixed orbit length: {:.3f} sec'.format(time.time()-t0))
        return df_PID

    def calibrate_fixedPID(self, test_PID, L_range=[128.7, 129.5, 30], in_global=False, save_PID_dataframe=True):
        '''
        the method to loop the PID results with a range of testing orbit length when you know the PID result of one peak in the experimental spectrum
        test_PID:       with a format of (ion, isomeric state, peakloc in experimental spectrum, harmonic) to start the PID loop
        L_range:        with a format of [L_start, L_end, num of L points]
        in_global:      if this method be inserted in calibrate_global, then True, default False
        '''
        t0 = time.time()
        L_range = np.linspace(L_range[0], L_range[1], num=L_range[-1], endpoint=True)
        df_PID_best = pd.DataFrame(columns=['peakloc', 'ID', 'harmonic', 'Brho_test', 'Brho_est', 'C_test', 'C_est', 'reduced_chisqr', 'addition_chisqr', 'skip_ionNum', 'corr_ionNum'])
        test_ion, test_isomeric, test_peakloc, test_harmonic = test_PID
        df_i = 0 # PID dataframe writing line number
        if not in_global:
            sess = WolframLanguageSession()
        else:
            sess = self.sess
        for _init_L in L_range:
            self.update_L_CSRe(_init_L)
            _Brho = self.calibrate_peak_loc(*test_PID)
            self.calibrate_Brho(_Brho)
            temp = self.cur.execute("SELECT ION, ISOMERIC, HARMONIC, PEAKLOC, MASS, Q FROM ISOCHRONOUSION WHERE YIELD >= 1").fetchall()
            _ion_data = pd.DataFrame({'ion': [_temp[0] for _temp in temp],
                    'isomeric': [_temp[1] for _temp in temp],
                    'harmonic': [_temp[2] for _temp in temp], 
                    'peakloc': [_temp[3] for _temp in temp],
                    'mass': [_temp[4] for _temp in temp],
                    'Q': [_temp[5] for _temp in temp]})
            chi_sqr, skip_ionNum, chi_num = 0, 0, 0
            # solve equation to get minimum of chiSqr
            expr_temp_00 = '' # inital expression
            expr_temp_01 = '' 
            expr_temp_02 = '' 
            expr_temp_03 = '' 
            expr_temp_04 = ''
            expr_temp_10 = ''
            expr_temp_11 = ''
            expr_temp_12 = ''
            for ii_spec, irow_spec in self.sweep_spec_peaks.iterrows():
                _ind = np.abs(_ion_data['peakloc']-irow_spec['exp_freq']).idxmin()
                if np.abs(_ion_data['peakloc'][_ind] - irow_spec['exp_freq']) <= 1.5/2/np.sqrt(2*np.log(2))*irow_spec['peak_width']: 
                    mass, Q, exp_freq = _ion_data['mass'][_ind], _ion_data['Q'][_ind], (irow_spec['exp_freq']+self.cen_freq*1e3)/_ion_data['harmonic'][_ind]*1e3
                    expr_temp_00 += ''' + {:.16f}^2 * {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2)^(3/2) '''.format(mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c)
                    expr_temp_01 += ''' + {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2 )^(1/2) '''.format(exp_freq, mass/Q/self.e*self.u2kg, self.c)
                    expr_temp_02 += ''' + ( {:.16f} / ( {:.16f}^2 / x^2 + 1 / {:}^2 ) )^2 '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c)
                    expr_temp_03 += ''' + 1 / ( {:.16f}^2 / x^2 + 1 / {:.16f}^2 ) '''.format(mass/Q/self.e*self.u2kg, self.c)
                    expr_temp_04 += ''' + ({:} - 1 / ( {:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(1/2) / CC)^2 '''.format(exp_freq, mass/Q/self.e*self.u2kg, self.c)
                    expr_temp_10 += ''' + 6 / CC^4 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2) - 4 * {:.16f} / CC^3 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(1/2) '''.format(mass/Q/self.e*self.u2kg, self.c, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_CC = A
                    expr_temp_11 += ''' - 4 * {:.16f}^2 / Brho^3 / CC^3 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^2 + 2 * {:.16f}^2 / Brho^3 / CC^2 * {:.16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(3/2) '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_CBrho = B
                    expr_temp_12 += ''' - 4 * {:.16f}^4 / Brho^6 / CC^2 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^3 + 6 * {:.16f}^4 / Brho^6 / CC * {:.16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(5/2) + 6 * {:.16f}^2 / Brho^4 / CC * {:16f} / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^(3/2) - 6 * {:.16f}^2 / Brho^4 / CC^2 / ({:.16f}^2 / Brho^2 + 1 / {:.16f}^2)^2 '''.format(mass/Q/self.e*self.u2kg, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c, mass/Q/self.e*self.u2kg, exp_freq, mass/Q/self.e*self.u2kg, self.c) # g_BrhoBrho = C
                    chi_num += 1
                else:
                    skip_ionNum += 1
                    continue
            expr_temp = '''result = FindRoot[({:}) / ({:}) - ({:}) / ({:}) == 0, {{x, {:}}}];
                            Brho = x /. result;
                            CC = ({:}) / ({:}) /. result;
                            ChiSqr = ({:}) / {:};
                            judgeA = {:} > 0;
                            judgeABC = ({:}) * ({:}) - ({:})^2 > 0;'''.format(expr_temp_00[2:], expr_temp_01[2:], expr_temp_02[2:], expr_temp_03[2:], _Brho, expr_temp_03[2:], expr_temp_01[2:], expr_temp_04[2:], chi_num, expr_temp_10[2:], expr_temp_10[2:], expr_temp_12, expr_temp_11)
            if chi_num > 2:
                sess.evaluate(wlexpr(expr_temp))
                iid_Brho = sess.evaluate(Global.Brho)
                iid_L = sess.evaluate(Global.CC)
                iid_ChiSqr = sess.evaluate(Global.ChiSqr)
                iid_judgeA = sess.evaluate(Global.judgeA)
                iid_judgeABC = sess.evaluate(Global.judgeABC)
                print('peakloc: {:}, id: {:}({:}), harmonic: {:}, Brho: {:.7f} Tm, C: {:.4f}\nChiSqr: {:.5f}, A>0: {:}, AC-B^2>0: {:}, Reduced_ChiSqr: {:}, Additional_ReChiSqr: {:}'.format(test_peakloc, test_ion, test_isomeric,  test_harmonic, iid_Brho, iid_L, iid_ChiSqr, iid_judgeA, iid_judgeABC, iid_ChiSqr/(chi_num-1), iid_ChiSqr/(chi_num-1)*skip_ionNum**2/chi_num**2))
                item_list = [test_peakloc, "{:}({:})".format(test_ion, test_isomeric), test_harmonic, _Brho, iid_Brho, _init_L, iid_L, iid_ChiSqr/(chi_num-1), iid_ChiSqr/(chi_num-1)*skip_ionNum**2/chi_num**2, skip_ionNum, chi_num]
                df_PID_best.loc[df_i] = item_list
                df_i += 1
            else:
                pass
        
        if not in_global:
            sess.stop()

        if save_PID_dataframe:
            df_PID_best.to_csv('./PID_{:}_loop_bestBrhoC_{:}.csv'.format(test_ion,self.iid_file), encoding='utf-8', index=True)
            print('iid results is saved in ./PID_{:}_loop_bestBrhoC_{:}.csv'.format(test_ion,self.iid_file))
        print('time for calibrating with fixed PID: {:.3f} sec'.format(time.time()-t0))
        return df_PID_best
    

    def calibrate_global(self, init_Brho, delta_Brho, L_range, save_PID_dataframe):
        '''
        the method to get the best Brho-C for the PID result
        init_Brho:      the inital testing Brho, [Tm], default None using the same Brho from .lpp setting
        delta_Brho:     the Brho loop range, |(testing Brho - init Brho)/init Brho| <= delta_Brho
        L_range:        with a format of [L_start, L_end, num of L points]
        '''
        # call Wolfram Kernel
        self.sess = WolframLanguageSession()
        try: 
            df_PID = self.calibrate_fixedC(self.L_CSRe, init_Brho, delta_Brho, True, save_PID_dataframe)
        except:
            self.sess.stop()
            return

        df_PID.sort_values(['addition_chisqr'], ascending=True, inplace=True)
        df_PID = df_PID.reset_index(drop=True)
        test_ion, test_isomeric, test_harmonic, test_peakloc = df_PID['ID'].values[0][:-3], int(df_PID['ID'].values[0][-2]), df_PID['harmonic'].values[0], df_PID['peakloc'].values[0]
        print("test with {:}({:}), h={:} at {:} kHz".format(test_ion, test_isomeric, test_harmonic, test_peakloc))
        try:
            df_PID_best = self.calibrate_fixedPID((test_ion, test_isomeric, test_peakloc, test_harmonic), L_range, True, save_PID_dataframe)
        except:
            print("something wrong with your setting for the loop, please check it!")
            self.sess.stop()
            return
        self.sess.stop()

        return df_PID_best


    def display_iidResult(self, Brho, L_CSRe, name):
        self.update_L_CSRe(L_CSRe)
        self.calibrate_Brho(Brho)
        temp = self.cur.execute("SELECT ION, ISOMERIC, HARMONIC, PEAKLOC FROM ISOCHRONOUSION WHERE YIELD >= 1").fetchall()
        _ion_data = pd.DataFrame({'ion': [_temp[0] for _temp in temp],
            'isomeric': [_temp[1] for _temp in temp],
            'harmonic': [_temp[2] for _temp in temp], 
            'peakloc': [_temp[3] for _temp in temp]})
        iid_ion, iid_isomeric, iid_harmonic = [], [], []
        for i_spec, row_spec in self.peak_file.iterrows():
            _ind = np.abs(_ion_data['peakloc']-row_spec['exp_freq']).idxmin()
            if np.abs(_ion_data['peakloc'][_ind] - row_spec['exp_freq']) <= 3/2/np.sqrt(2*np.log(2))*row_spec['peak_width']:
                iid_ion.append(_ion_data['ion'][_ind])
                iid_isomeric.append(_ion_data['isomeric'][_ind])
                iid_harmonic.append(_ion_data['harmonic'][_ind])
                print('peakloc {:} kHz: {:}({:}), {:}'.format(row_spec['exp_freq']+self.cen_freq*1e3, iid_ion[-1], iid_isomeric[-1], iid_harmonic[-1]))
            else:
                iid_ion.append('null')
                iid_isomeric.append('0')
                iid_harmonic.append(0)
                print('peakloc {:}: not in .lpp files'.format(row_spec['exp_freq']))
        pd.DataFrame({'peakloc': self.peak_file['exp_freq'] + self.cen_freq*1e3, 'peak_height': self.peak_file['peak_height'], 'peak_width': self.peak_file['peak_width'], 'ion': iid_ion, 'isomeric': iid_isomeric, 'harmonic': iid_harmonic}).to_csv('./PID_display_{:}_{:}.csv'.format(name, self.iid_file))
        print('iid results were saved in ./PID_display_{:}_{:}.csv'.format(name, self.iid_file))




if __name__ == "__main__":
    sw_0 = SWEEP_IID('188Hf.csv', 'New_TEline2022-ESR-188Hf-208Pb-nostripper_509MeV-v2.lpp', 411.742, 50000, 65536, 1.4, 108.34)
    sw_0.calibrate_global(init_Brho=7.9183, delta_Brho=0.001, L_range=[108.25, 108.58, 40], save_PID_dataframe=True)
    #sw_1 = SWEEP_IID('186Hf-new.csv', 'New_TEline2022-ESR-186Hf-208Pb-nostripper_510MeV.lpp', 411.843, 50000, 65536, 1.4, 108.34)
    #sw_1.calibrate_global(init_Brho=7.8369, delta_Brho=0.001, L_range=[108.25, 108.58, 40], save_PID_dataframe=True)
    #SWEEP_IID('192Pb-summed.csv', 'New_TEline2022-ESR-192Pb-208Pb-nostripper_524MeV.lpp', 411.757, 6250, 131072, 1.4, init_Brho=None, delta_Brho=0.01, L_CSRe=108.34)
    #SWEEP_IID('194Pb.csv', 'New_TEline2022-ESR-194Pb-208Pb-nostripper_524MeV.lpp', 410.535, 50000, 65536, 2.47, 1.4, init_Brho=None, delta_Brho=0.01, L_CSRe=108.34)


