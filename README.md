# BDI Fear & Greed Index — Réplica CNN Mejorada

App Streamlit con la réplica del **Fear & Greed Index de CNN**, mejorada y con la identidad visual de **BDI Consultora Patrimonial Integral**.

## Características

- 7 componentes (momentum, breadth, strength, PCR proxy, volatilidad, safe haven, junk bond)
- - Normalización Z-score rolling 252d + sigmoide
  - - VIX en valor real (no percentil)
    - - Cache 30 min
      - - Sección educativa con fórmulas LaTeX
       
        - ## Probar local
       
        - ```bash
          pip install -r requirements.txt
          streamlit run app.py
          ```

          ## Deploy

          Desplegado en [Streamlit Community Cloud](https://share.streamlit.io).

          ---

          BDI Consultora Patrimonial Integral · Material educativo, no es asesoramiento financiero.
          
