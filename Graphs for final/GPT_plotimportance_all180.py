import re
import matplotlib.pyplot as plt
import pandas as pd

# Provided data
data = '''
LDA Feature 71: Importance 0.0900
MDI Feature 36: Importance 0.010137435453469568
LDA Feature 30: Importance 0.0585
MDI Feature 35: Importance 0.006129685676634276
LDA Feature 59: Importance 0.0459
MDI Feature 5: Importance 0.006159956636404343
LDA Feature 32: Importance 0.0435
MDI Feature 34: Importance 0.0066386899422651805
LDA Feature 65: Importance 0.0371
MDI Feature 8: Importance 0.009758934427033314
LDA Feature 35: Importance 0.0305
MDI Feature 33: Importance 0.03327848868642738
LDA Feature 24: Importance 0.0302
MDI Feature 20: Importance 0.006967266639444883
LDA Feature 18: Importance 0.0295
MDI Feature 56: Importance 0.006281183077576758
LDA Feature 139: Importance 0.0293
MDI Feature 31: Importance 0.03183360359591047
LDA Feature 20: Importance 0.0225
MDI Feature 29: Importance 0.003962653427389858
LDA Feature 26: Importance 0.0200
MDI Feature 26: Importance 0.004178012650720987
LDA Feature 34: Importance 0.0195
MDI Feature 41: Importance 0.013819410415115141
LDA Feature 66: Importance 0.0191
MDI Feature 44: Importance 0.0011958984609159713
LDA Feature 67: Importance 0.0175
MDI Feature 62: Importance 0.0007380004980280451
LDA Feature 29: Importance 0.0169
MDI Feature 32: Importance 0.0022885962423251863
LDA Feature 177: Importance 0.0157
MDI Feature 11: Importance 0.001027095972681978
LDA Feature 127: Importance 0.0152
MDI Feature 65: Importance 0.0027998954128199195
LDA Feature 23: Importance 0.0147
MDI Feature 23: Importance 0.0017934992086973797
LDA Feature 70: Importance 0.0141
MDI Feature 30: Importance 0.006625314363556499
LDA Feature 133: Importance 0.0137
MDI Feature 47: Importance 0.005876062041187695
LDA Feature 143: Importance 0.0129
MDI Feature 39: Importance 0.021290272702217568
LDA Feature 54: Importance 0.0125
MDI Feature 59: Importance 0.0035568520732073296
LDA Feature 55: Importance 0.0121
MDI Feature 69: Importance 0.005163390810524439
LDA Feature 68: Importance 0.0109
MDI Feature 37: Importance 0.013176957200331572
LDA Feature 33: Importance 0.0109
MDI Feature 71: Importance 0.007816850106325338
LDA Feature 102: Importance 0.0105
MDI Feature 0: Importance 0.0058663793377335014
LDA Feature 69: Importance 0.0102
MDI Feature 4: Importance 0.018126546250943103
LDA Feature 60: Importance 0.0101
MDI Feature 72: Importance 0.004878754811863375
LDA Feature 22: Importance 0.0101
MDI Feature 113: Importance 0.007552452785401938
LDA Feature 47: Importance 0.0092
MDI Feature 134: Importance 0.020385049448332068
LDA Feature 43: Importance 0.0087
MDI Feature 42: Importance 0.012618592712234809
LDA Feature 106: Importance 0.0083
MDI Feature 127: Importance 0.021240666239922112
LDA Feature 28: Importance 0.0080
MDI Feature 24: Importance 0.014474613549221827
LDA Feature 171: Importance 0.0077
MDI Feature 133: Importance 0.021711178639352204
LDA Feature 165: Importance 0.0076
MDI Feature 116: Importance 0.032479033282067575
LDA Feature 104: Importance 0.0073
MDI Feature 77: Importance 0.033943401513183286
LDA Feature 58: Importance 0.0071
MDI Feature 60: Importance 0.04032458994710101
LDA Feature 137: Importance 0.0071
MDI Feature 28: Importance 0.010371873102385142
LDA Feature 27: Importance 0.0070
MDI Feature 128: Importance 0.006757516810734701
LDA Feature 64: Importance 0.0068
MDI Feature 6: Importance 0.012478438517643515
LDA Feature 25: Importance 0.0067
MDI Feature 115: Importance 0.003702187691394071
LDA Feature 179: Importance 0.0066
MDI Feature 70: Importance 0.017491038801871667
LDA Feature 142: Importance 0.0065
MDI Feature 38: Importance 0.008407640038235055
LDA Feature 96: Importance 0.0059
MDI Feature 67: Importance 0.006438769678682002
LDA Feature 141: Importance 0.0058
MDI Feature 54: Importance 0.016457530173234397
LDA Feature 63: Importance 0.0058
MDI Feature 3: Importance 0.0036904518490147964
LDA Feature 31: Importance 0.0057
MDI Feature 18: Importance 0.0037368430262873357
LDA Feature 90: Importance 0.0057
MDI Feature 43: Importance 0.012613121211974228
LDA Feature 62: Importance 0.0057
MDI Feature 147: Importance 0.0010137791284027854
LDA Feature 21: Importance 0.0056
MDI Feature 7: Importance 0.00044069137683810776
LDA Feature 103: Importance 0.0053
MDI Feature 109: Importance 0.001054818828919308
LDA Feature 7: Importance 0.0050
MDI Feature 90: Importance 0.0006911607327126412
LDA Feature 131: Importance 0.0049
MDI Feature 2: Importance 0.0008569338231292094
LDA Feature 56: Importance 0.0048
MDI Feature 1: Importance 0.0022469415341484973
LDA Feature 94: Importance 0.0044
MDI Feature 19: Importance 0.00671240313659383
LDA Feature 105: Importance 0.0042
MDI Feature 25: Importance 0.0043901812753215155
LDA Feature 53: Importance 0.0042
MDI Feature 101: Importance 0.021270915648064572
LDA Feature 175: Importance 0.0042
MDI Feature 61: Importance 0.004019350972547994
LDA Feature 57: Importance 0.0039
MDI Feature 95: Importance 0.003520606870334197
LDA Feature 98: Importance 0.0037
MDI Feature 78: Importance 0.01195702559824517
LDA Feature 37: Importance 0.0037
MDI Feature 92: Importance 0.007664701231493039
LDA Feature 178: Importance 0.0037
MDI Feature 22: Importance 0.0055713427112603
LDA Feature 12: Importance 0.0037
MDI Feature 98: Importance 0.0150260583763516
LDA Feature 129: Importance 0.0037
MDI Feature 131: Importance 0.0030985159052944627
LDA Feature 167: Importance 0.0035
MDI Feature 27: Importance 0.0041396521244036885
LDA Feature 36: Importance 0.0034
MDI Feature 68: Importance 0.013182636408381774
LDA Feature 100: Importance 0.0034
MDI Feature 149: Importance 0.004173676898091201
LDA Feature 97: Importance 0.0033
MDI Feature 96: Importance 0.006746550725256623
LDA Feature 61: Importance 0.0033
MDI Feature 80: Importance 0.004859828461940984
LDA Feature 176: Importance 0.0031
MDI Feature 114: Importance 0.011949632641720631
LDA Feature 130: Importance 0.0031
MDI Feature 55: Importance 0.006816364008054811
LDA Feature 136: Importance 0.0030
MDI Feature 107: Importance 0.010160782228983185
LDA Feature 140: Importance 0.0030
MDI Feature 10: Importance 0.009670225094533379
LDA Feature 92: Importance 0.0030
MDI Feature 66: Importance 0.004165687473533071
LDA Feature 173: Importance 0.0029
MDI Feature 73: Importance 0.003403012304805713
LDA Feature 99: Importance 0.0029
MDI Feature 64: Importance 0.0017440493700677834
LDA Feature 135: Importance 0.0027
MDI Feature 132: Importance 0.0022497406078857475
LDA Feature 107: Importance 0.0026
MDI Feature 79: Importance 0.007673144167184735
LDA Feature 101: Importance 0.0023
MDI Feature 170: Importance 0.005449031307747167
LDA Feature 119: Importance 0.0023
MDI Feature 57: Importance 0.0040883385561754524
LDA Feature 128: Importance 0.0022
MDI Feature 111: Importance 0.004605082204691573
LDA Feature 91: Importance 0.0022
MDI Feature 9: Importance 0.0011822619364869314
LDA Feature 10: Importance 0.0022
MDI Feature 137: Importance 0.0031950949501349953
LDA Feature 11: Importance 0.0021
MDI Feature 46: Importance 0.0037245656319652505
LDA Feature 19: Importance 0.0021
MDI Feature 83: Importance 0.0005089742520544706
LDA Feature 49: Importance 0.0021
MDI Feature 40: Importance 0.0005016230192785488
LDA Feature 95: Importance 0.0020
MDI Feature 100: Importance 0.0002963380490207531
LDA Feature 42: Importance 0.0020
MDI Feature 45: Importance 0.0003721833788722857
LDA Feature 164: Importance 0.0019
MDI Feature 21: Importance 0.00039118672584153197
LDA Feature 14: Importance 0.0019
MDI Feature 58: Importance 0.0011566507107787873
LDA Feature 41: Importance 0.0018
MDI Feature 152: Importance 0.006160759348693204
LDA Feature 163: Importance 0.0018
MDI Feature 74: Importance 0.003309243883717656
LDA Feature 8: Importance 0.0018
MDI Feature 119: Importance 0.005315701694247655
LDA Feature 166: Importance 0.0016
MDI Feature 110: Importance 0.0009433633461533055
LDA Feature 109: Importance 0.0014
MDI Feature 91: Importance 0.003299469586871289
LDA Feature 45: Importance 0.0014
MDI Feature 94: Importance 0.00546496581446977
LDA Feature 81: Importance 0.0013
MDI Feature 82: Importance 0.004665796319072676
LDA Feature 93: Importance 0.0012
MDI Feature 117: Importance 0.0031043954999405887
LDA Feature 152: Importance 0.0011
MDI Feature 97: Importance 0.005112450666526646
LDA Feature 169: Importance 0.0011
MDI Feature 164: Importance 0.0014458604431661782
LDA Feature 17: Importance 0.0011
MDI Feature 63: Importance 0.003696016666074398
LDA Feature 134: Importance 0.0011
MDI Feature 136: Importance 0.005788538709833194
LDA Feature 73: Importance 0.0010
MDI Feature 143: Importance 0.0008592196177622471
LDA Feature 82: Importance 0.0009
MDI Feature 118: Importance 0.0014337599462375284
LDA Feature 117: Importance 0.0009
MDI Feature 130: Importance 0.0011675056640816498
LDA Feature 151: Importance 0.0009
MDI Feature 135: Importance 0.001149948850955985
LDA Feature 9: Importance 0.0009
MDI Feature 16: Importance 0.0015164754510335581
LDA Feature 172: Importance 0.0009
MDI Feature 126: Importance 0.004293961892202879
LDA Feature 44: Importance 0.0008
MDI Feature 167: Importance 0.0012386015986760978
LDA Feature 0: Importance 0.0008
MDI Feature 171: Importance 0.006205748748406747
LDA Feature 15: Importance 0.0008
MDI Feature 129: Importance 0.00336317254020133
LDA Feature 51: Importance 0.0007
MDI Feature 155: Importance 0.003992196216503325
LDA Feature 40: Importance 0.0007
MDI Feature 173: Importance 0.0019820665774124375
LDA Feature 79: Importance 0.0007
MDI Feature 14: Importance 0.008658728708512786
LDA Feature 159: Importance 0.0007
MDI Feature 76: Importance 0.004581479446236905
LDA Feature 78: Importance 0.0006
MDI Feature 53: Importance 0.0069140350229491224
LDA Feature 149: Importance 0.0006
MDI Feature 166: Importance 0.007795272068412284
LDA Feature 126: Importance 0.0006
MDI Feature 165: Importance 0.0031573660690591893
LDA Feature 113: Importance 0.0006
MDI Feature 169: Importance 0.0029667501434203192
LDA Feature 115: Importance 0.0006
MDI Feature 112: Importance 0.003387287780171815
LDA Feature 154: Importance 0.0005
MDI Feature 154: Importance 0.0003946107331637047
LDA Feature 13: Importance 0.0005
MDI Feature 153: Importance 0.0003930810495162423
LDA Feature 72: Importance 0.0005
MDI Feature 148: Importance 0.000315817647610581
LDA Feature 39: Importance 0.0004
MDI Feature 172: Importance 0.0005972322190293846
LDA Feature 121: Importance 0.0004
MDI Feature 17: Importance 0.00023915892477739055
LDA Feature 170: Importance 0.0004
MDI Feature 163: Importance 0.001451102727190575
LDA Feature 147: Importance 0.0004
MDI Feature 146: Importance 0.0027927270611131246
LDA Feature 38: Importance 0.0004
MDI Feature 75: Importance 0.007831401077280545
LDA Feature 160: Importance 0.0004
MDI Feature 151: Importance 0.007167917378705527
LDA Feature 75: Importance 0.0004
MDI Feature 106: Importance 0.0024570271390843253
LDA Feature 46: Importance 0.0004
MDI Feature 125: Importance 0.0028871452888771442
LDA Feature 80: Importance 0.0004
MDI Feature 99: Importance 0.0050037278834008485
LDA Feature 174: Importance 0.0004
MDI Feature 103: Importance 0.0041184730457845455
LDA Feature 155: Importance 0.0003
MDI Feature 145: Importance 0.007800837954032224
LDA Feature 111: Importance 0.0003
MDI Feature 168: Importance 0.0084736807896046
LDA Feature 6: Importance 0.0003
MDI Feature 108: Importance 0.0028275524077106735
LDA Feature 3: Importance 0.0003
MDI Feature 162: Importance 0.003050015403707372
LDA Feature 162: Importance 0.0003
MDI Feature 12: Importance 0.0038380333091668577
LDA Feature 158: Importance 0.0003
MDI Feature 81: Importance 0.0010769548864848026
LDA Feature 77: Importance 0.0003
MDI Feature 104: Importance 0.0006811046364230817
LDA Feature 16: Importance 0.0003
MDI Feature 89: Importance 0.0011235023081275532
LDA Feature 85: Importance 0.0003
MDI Feature 105: Importance 0.0009054740267450594
LDA Feature 4: Importance 0.0003
MDI Feature 140: Importance 0.0008549813161542756
LDA Feature 1: Importance 0.0003
MDI Feature 144: Importance 0.0030233165761250212
LDA Feature 83: Importance 0.0003
MDI Feature 138: Importance 0.0011225200465627976
LDA Feature 5: Importance 0.0003
MDI Feature 50: Importance 0.0013193969328063597
LDA Feature 52: Importance 0.0003
MDI Feature 15: Importance 0.0017530154691007655
LDA Feature 87: Importance 0.0003
MDI Feature 177: Importance 0.006359702388586597
LDA Feature 114: Importance 0.0003
MDI Feature 150: Importance 0.0018743504283509841
LDA Feature 125: Importance 0.0002
MDI Feature 48: Importance 0.004785640054158147
LDA Feature 153: Importance 0.0002
MDI Feature 93: Importance 0.0010167319930374562
LDA Feature 148: Importance 0.0002
MDI Feature 175: Importance 0.0017341118845399153
LDA Feature 145: Importance 0.0002
MDI Feature 141: Importance 0.003475020640208426
LDA Feature 118: Importance 0.0002
MDI Feature 102: Importance 0.0019031560812447553
LDA Feature 157: Importance 0.0002
MDI Feature 52: Importance 0.0019302016449233806
LDA Feature 86: Importance 0.0002
MDI Feature 142: Importance 0.002430499212160679
LDA Feature 110: Importance 0.0002
MDI Feature 13: Importance 0.00023315969606712666
LDA Feature 123: Importance 0.0002
MDI Feature 51: Importance 0.0003582639281015547
LDA Feature 84: Importance 0.0001
MDI Feature 139: Importance 0.0003234996226919036
LDA Feature 108: Importance 0.0001
MDI Feature 179: Importance 0.00048092231899179945
LDA Feature 48: Importance 0.0001
MDI Feature 178: Importance 0.00037776443556881013
LDA Feature 168: Importance 0.0001
MDI Feature 174: Importance 0.0004128861563609286
LDA Feature 89: Importance 0.0001
MDI Feature 123: Importance 0.001237925673087156
LDA Feature 120: Importance 0.0001
MDI Feature 176: Importance 0.0017561923471443167
LDA Feature 112: Importance 0.0001
MDI Feature 84: Importance 0.0031015816504166922
LDA Feature 50: Importance 0.0001
MDI Feature 85: Importance 0.0020437640302018094
LDA Feature 124: Importance 0.0001
MDI Feature 159: Importance 0.0021245497507809737
LDA Feature 74: Importance 0.0001
MDI Feature 49: Importance 0.0027623373415047903
LDA Feature 156: Importance 0.0001
MDI Feature 161: Importance 0.0012761680486846824
LDA Feature 88: Importance 0.0001
MDI Feature 120: Importance 0.002005528993104284
LDA Feature 146: Importance 0.0001
MDI Feature 121: Importance 0.004066513663714954
LDA Feature 122: Importance 0.0001
MDI Feature 88: Importance 0.002675680232053208
LDA Feature 132: Importance 0.0001
MDI Feature 160: Importance 0.0018694853764083055
LDA Feature 144: Importance 0.0001
MDI Feature 87: Importance 0.002323313914892628
LDA Feature 76: Importance 0.0000
MDI Feature 157: Importance 0.0006281321761210557
LDA Feature 161: Importance 0.0000
MDI Feature 158: Importance 0.000935066258187774
LDA Feature 116: Importance 0.0000
MDI Feature 122: Importance 0.000557621260674067
LDA Feature 150: Importance 0.0000
MDI Feature 86: Importance 0.0010199631562031383
LDA Feature 138: Importance 0.0000
MDI Feature 124: Importance 0.0006331402163175491
LDA Feature 2: Importance 0.0000
MDI Feature 156: Importance 0.0006347633386488826
'''

# Extract data using regex
pattern = re.compile(r'(LDA|MDI) Feature (\d+): Importance ([\d.]+)')
matches = pattern.findall(data)

# Convert to DataFrame
df = pd.DataFrame(matches, columns=['Method', 'Feature', 'Importance'])
df['Feature'] = df['Feature'].astype(int)
df['Importance'] = df['Importance'].astype(float)

# Define feature to location mapping
feature_to_location = {
    **dict.fromkeys(range(0, 30), 'hand'),
    **dict.fromkeys(range(30, 60), 'lower arm'),
    **dict.fromkeys(range(60, 90), 'upper arm'),
    **dict.fromkeys(range(90, 120), 'shoulder'),
    **dict.fromkeys(range(120, 150), 'sternum'),
}

# Add location to DataFrame
df['Location'] = df['Feature'].map(feature_to_location)

# Split DataFrame into LDA and MDI
lda_df = df[df['Method'] == 'LDA']
mdi_df = df[df['Method'] == 'MDI']

# Aggregate importance by location
lda_agg = lda_df.groupby('Location')['Importance'].sum().reset_index()
mdi_agg = mdi_df.groupby('Location')['Importance'].sum().reset_index()

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(lda_agg['Location'], lda_agg['Importance'], color='blue')
plt.title('LDA Feature Importance by Location')
plt.xlabel('Location')
plt.ylabel('Total Importance')

plt.subplot(1, 2, 2)
plt.bar(mdi_agg['Location'], mdi_agg['Importance'], color='green')
plt.title('MDI Feature Importance by Location')
plt.xlabel('Location')
plt.ylabel('Total Importance')

plt.tight_layout()
plt.show()
