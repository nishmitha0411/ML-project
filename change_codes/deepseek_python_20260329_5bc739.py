    # Check image quality
    quality_ok, quality_message = check_image_quality(img_array)
    if not quality_ok:
        st.error(quality_message)
        st.info("💡 Tips: Use a well-lit, clear photo of your face")
        st.stop()  # Stop processing if quality is bad
    else:
        st.success(quality_message)
