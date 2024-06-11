import pandas as pd
import matplotlib.pyplot as plt

# Data for LDA and MDI
data_lda = {
    "Feature Importance": [0.0900, 0.0585, 0.0459, 0.0435, 0.0371, 0.0305, 0.0302, 0.0295, 0.0293, 
                           0.0225, 0.0200, 0.0195, 0.0191, 0.0175, 0.0169, 0.0157, 0.0152, 0.0147, 
                           0.0141, 0.0137, 0.0129, 0.0125, 0.0121, 0.0109, 0.0109, 0.0105, 0.0102, 
                           0.0101, 0.0101, 0.0092],
    "Feature Name": ["lower arm_slope_z-rot", "hand_mean_x-acc", "sternum_mean_z-rot", "hand_mean_z-acc", 
                     "hand_slope_z-rot", "hand_mean_z-rot", "sternum_rms_x-acc", "shoulder_rms_x-acc", 
                     "shoulder_min_y-acc", "shoulder_rms_z-acc", "sternum_rms_z-acc", "hand_mean_y-rot", 
                     "lower arm_slope_x-acc", "lower arm_slope_y-acc", "sternum_rms_z-rot", "sternum_std_x-rot", 
                     "lower arm_min_y-acc", "shoulder_rms_z-rot", "lower arm_slope_y-rot", "upper arm_min_y-acc", 
                     "shoulder_min_z-rot", "sternum_mean_x-acc", "sternum_mean_y-acc", "lower arm_slope_z-acc", 
                     "hand_mean_x-rot", "upper arm_max_x-acc", "lower arm_slope_x-rot", "hand_slope_x-acc", 
                     "shoulder_rms_y-rot", "upper arm_mean_z-rot"]
}

data_mdi = {
    "Feature Importance": [0.0101, 0.0061, 0.0062, 0.0066, 0.0098, 0.0333, 0.0070, 0.0063, 0.0318, 
                           0.0040, 0.0042, 0.0138, 0.0012, 0.0007, 0.0023, 0.0010, 0.0028, 0.0018, 
                           0.0066, 0.0059, 0.0213, 0.0036, 0.0052, 0.0132, 0.0078, 0.0059, 0.0181, 
                           0.0049, 0.0076, 0.0204],
    "Feature Name": ["lower arm_mean_x-acc", "hand_mean_z-rot", "hand_rms_z-rot", "hand_mean_y-rot", 
                     "lower arm_rms_z-acc", "hand_mean_x-rot", "shoulder_rms_z-acc", "sternum_mean_z-acc", 
                     "hand_mean_y-acc", "sternum_rms_z-rot", "sternum_rms_z-acc", "lower arm_mean_z-rot", 
                     "upper arm_mean_z-acc", "hand_slope_z-acc", "hand_mean_z-acc", "lower arm_rms_z-rot", 
                     "hand_slope_z-rot", "shoulder_rms_z-rot", "hand_mean_x-acc", "upper arm_mean_z-rot", 
                     "lower arm_mean_x-rot", "sternum_mean_z-rot", "lower arm_slope_x-rot", "lower arm_mean_y-acc", 
                     "lower arm_slope_z-rot", "hand_rms_x-acc", "hand_rms_y-rot", "upper arm_slope_x-acc", 
                     "shoulder_max_z-rot", "upper arm_min_z-acc"]
}

# Creating DataFrames
df_lda = pd.DataFrame(data_lda)
df_mdi = pd.DataFrame(data_mdi)

# Extract locations from feature names
def extract_location(feature_name):
    return feature_name.split('_')[0]

# Add locations to dataframes
df_lda['Location'] = df_lda['Feature Name'].apply(extract_location)
df_mdi['Location'] = df_mdi['Feature Name'].apply(extract_location)

# Group by Location and sum importance
lda_location_importance = df_lda.groupby('Location')['Feature Importance'].sum().reset_index()
mdi_location_importance = df_mdi.groupby('Location')['Feature Importance'].sum().reset_index()

# Sort values by importance
lda_location_importance = lda_location_importance.sort_values(by='Feature Importance', ascending=False)
mdi_location_importance = mdi_location_importance.sort_values(by='Feature Importance', ascending=False)

# Convert to arrays for the plots
lda_locations = lda_location_importance['Location'].values
lda_importances = lda_location_importance['Feature Importance'].values
mdi_locations = mdi_location_importance['Location'].values
mdi_importances = mdi_location_importance['Feature Importance'].values

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# LDA Plot
axes[0].bar(lda_locations, lda_importances, color='blue')
axes[0].set_title('LDA Feature Importance by Location')
axes[0].set_xlabel('Location')
axes[0].set_ylabel('Total Feature Importance')
axes[0].set_xticklabels(lda_locations, size = 15)

# MDI Plot
axes[1].bar(mdi_locations, mdi_importances, color='green')
axes[1].set_title('MDI Feature Importance by Location')
axes[1].set_xlabel('Location')
axes[1].set_ylabel('Total Feature Importance')
axes[1].set_xticklabels(mdi_locations, size = 15)


plt.tight_layout()
plt.show()