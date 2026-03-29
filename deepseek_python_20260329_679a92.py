    # =========================
    # 📥 DOWNLOAD REPORT
    # =========================
    st.markdown("---")
    report_text = f"""
PERSONAL COLOR ANALYSIS REPORT
================================
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS
--------
Skin Undertone: {prediction}
Seasonal Type: {season}
Confidence: {confidence:.1f}%

DETECTED COLORS
---------------
Red: {r}
Green: {g}
Blue: {b}

RECOMMENDATIONS
---------------
Best Colors: Based on your {prediction} undertone
Jewelry: {'Gold' if prediction == 'Warm' else 'Silver' if prediction == 'Cool' else 'Both gold and silver'}
    """
    
    st.download_button(
        label="📥 Download My Report",
        data=report_text,
        file_name=f"color_analysis_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )