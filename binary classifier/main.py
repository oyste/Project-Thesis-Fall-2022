from extract_feature_functions import *

x = np.load('x.npy')

#x_wsn = get_wsn_feats(x)

#np.save('x_wsn_8_1_J_10_new_T.npy', x_wsn)

#x_altp, x_wsn, x_mfcc, y = np.load('x_altp.npy'), np.load('x_wsn.npy'), np.load('x_mfcc.npy'), np.load('y.npy')

#print('x_altp:',x_altp.shape,'x_wsn', x_wsn.shape,'x_mfcc:',x_mfcc.shape)
#np.save('x_all.npy', np.hstack([x_wsn, x_altp, x_mfcc]))

x_altp = get_altp_feats3(x)
np.save('x_altp_L_5.npy', x_altp)



