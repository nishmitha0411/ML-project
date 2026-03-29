                # =========================
                # 🎨 COLOR PALETTE
                # =========================
                st.subheader("🎨 Your Color Palette")
                
                # Generate palette
                palette = generate_color_palette(prediction)
                
                # Display palette as colored boxes
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: {palette['Primary']['hex']}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                        <b>Primary</b><br>
                        {palette['Primary']['name']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: {palette['Secondary']['hex']}; padding: 20px; border-radius: 10px; text-align: center; color: {'black' if palette['Secondary']['name'] == 'Mustard' else 'white'};">
                        <b>Secondary</b><br>
                        {palette['Secondary']['name']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background-color: {palette['Accent']['hex']}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                        <b>Accent</b><br>
                        {palette['Accent']['name']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="background-color: {palette['Neutral']['hex']}; padding: 20px; border-radius: 10px; text-align: center; color: {'black' if palette['Neutral']['name'] == 'Warm Beige' else 'white'};">
                        <b>Neutral</b><br>
                        {palette['Neutral']['name']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show hex codes
                st.markdown("### 📋 Color Codes")
                st.markdown(f"""
                | Color | Name | Hex Code | RGB |
                |-------|------|----------|-----|
                | 🟠 Primary | {palette['Primary']['name']} | {palette['Primary']['hex']} | {palette['Primary']['rgb']} |
                | 🟡 Secondary | {palette['Secondary']['name']} | {palette['Secondary']['hex']} | {palette['Secondary']['rgb']} |
                | 🟢 Accent | {palette['Accent']['name']} | {palette['Accent']['hex']} | {palette['Accent']['rgb']} |
                | ⚪ Neutral | {palette['Neutral']['name']} | {palette['Neutral']['hex']} | {palette['Neutral']['rgb']} |
                """)
