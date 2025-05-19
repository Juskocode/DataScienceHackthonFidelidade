# DataScienceHackthonFidelidade


## Sloth's Input
- Some variables doesn't have any or very low variance , so they can be deleted
- Some variables have an High Missing Value Rate, delete
- Some variables have high skewness and/or kurtosis
  
 === Recommended Transformations ===

For highly skewed columns:
1. Log transformation: For positive, right-skewed data
   Example: df['col_log'] = np.log1p(df['col'])
2. Square root transformation: For moderately right-skewed data
   Example: df['col_sqrt'] = np.sqrt(df['col'])
3. Box-Cox transformation: Automatically finds optimal power transformation
   Example: from scipy import stats
            df['col_boxcox'], _ = stats.boxcox(df['col'] + 1)
4. Yeo-Johnson transformation: Works for both positive and negative values
   Example: from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson')
            df['col_yj'] = pt.fit_transform(df[['col']])

Detailed distribution statistics saved to 'distribution_analysis.csv'
