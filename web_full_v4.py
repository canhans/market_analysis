import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1) ORTAK YARDIMCI
# ============================================================

def get_unit_column(unit_key: str) -> str:
    if unit_key == "m2":
        return "value_m2"
    elif unit_key == "USD":
        return "value_usd"
    else:
        raise ValueError(f"Bilinmeyen birim: {unit_key}")


# ============================================================
# 2) MİMARİ – TAM VERİLERİ (TAM.xlsx)
# ============================================================

@st.cache_data
def load_tam_yearly(excel_path: str) -> pd.DataFrame:
    """
    'Dis Cephe' ve 'İc Cephe' sheet'lerini okuyup:
    year | region | facade | metric_type | value_m2 | value_usd
    formatında döner.
    """

    def reshape_main(sheet_name: str, facade_label: str) -> pd.DataFrame:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

        header = df.iloc[1].tolist()
        data = df.iloc[2:, :len(header)].copy()
        data.columns = header

        data = data[pd.to_numeric(data["Yıllar"], errors="coerce").notna()]
        data["Yıllar"] = data["Yıllar"].astype(int)

        data_long = data.melt(
            id_vars=["Yıllar"],
            value_vars=["Avrupa + Rusya", "Orta Doğu", "Kuzey Amerika"],
            var_name="region_raw",
            value_name="value_m2",
        )

        region_map = {
            "Avrupa + Rusya": "Europe+Russia",
            "Orta Doğu": "MEA+Turkey",
            "Kuzey Amerika": "North America",
        }
        data_long["region"] = data_long["region_raw"].map(region_map).fillna(
            data_long["region_raw"]
        )

        data_long["facade"] = facade_label
        data_long["metric_type"] = "TAM"
        data_long["value_m2"] = pd.to_numeric(data_long["value_m2"], errors="coerce")
        data_long["value_usd"] = data_long["value_m2"]  # şimdilik aynı

        data_long = data_long.rename(columns={"Yıllar": "year"})
        return data_long[
            ["year", "region", "facade", "metric_type", "value_m2", "value_usd"]
        ]

    dis_long = reshape_main("Dis Cephe", "Dış Cephe")
    ic_long = reshape_main("İc Cephe", "İç Cephe")

    df_main = pd.concat([dis_long, ic_long], ignore_index=True)
    return df_main


@st.cache_data
def load_tam_segment_distributions(excel_path: str):
    """
    'Dis_cephe_dagilim' ve 'İc_Cephe_Dagilim' sheet'lerini
    region | segment | value formatına dönüştürür.
    """
    df_dis = pd.read_excel(excel_path, sheet_name="Dis_Cephe_Dagilim")
    df_ic = pd.read_excel(excel_path, sheet_name="Ic_Cephe_Dagilim")

    segments = df_dis.iloc[0, 1:-1].tolist()

    rows_dis = []
    regions_dis = df_dis.iloc[1:-1, 0].tolist()
    for i, region in enumerate(regions_dis):
        row_values = df_dis.iloc[i + 1, 1:-1].tolist()
        for seg, val in zip(segments, row_values):
            rows_dis.append([region, seg, val])
    df_dis_long = pd.DataFrame(rows_dis, columns=["region", "segment", "value"])

    rows_ic = []
    regions_ic = df_ic.iloc[1:-1, 0].tolist()
    for i, region in enumerate(regions_ic):
        row_values = df_ic.iloc[i + 1, 1:-1].tolist()
        for seg, val in zip(segments, row_values):
            rows_ic.append([region, seg, val])
    df_ic_long = pd.DataFrame(rows_ic, columns=["region", "segment", "value"])

    return df_dis_long, df_ic_long


# ============================================================
# 3) MİMARİ – SAM VERİLERİ (SAM.xlsx)
# ============================================================

@st.cache_data
def load_sam_data(excel_path: str):
    """
    SAM.xlsx içeriğini tek noktadan okur.
    Sheet'ler:
      - PDLC_Regional
      - SPD_Regional
      - Total_Regional
      - Technology_Total
      - PDLC_Dis_Cephe_Yillik
      - PDLC_Ic_Cephe_Yillik
      - SPD_Dis_Cephe_Yillik
    """

    # --- Bölgesel (PDLC / SPD / Total) ---
    def reshape_regional(sheet_name: str, tech_label: str) -> pd.DataFrame:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        header = raw.iloc[1].tolist()
        data = raw.iloc[2:, :len(header)].copy()
        data.columns = header

        data = data[pd.to_numeric(data["Yıllar"], errors="coerce").notna()]
        data["Yıllar"] = data["Yıllar"].astype(int)

        region_cols = [
            c for c in data.columns if c != "Yıllar" and "Toplam" not in c
        ]

        df_long = data.melt(
            id_vars=["Yıllar"],
            value_vars=region_cols,
            var_name="region_raw",
            value_name="value",
        )

        df_long["region_label"] = (
            df_long["region_raw"]
            .str.replace(r" \(m2\)", "", regex=True)
            .str.strip()
        )
        region_map = {
            "Avrupa + Rusya": "Europe+Russia",
            "Avrupa+Rusya": "Europe+Russia",
            "Orta Doğu": "MEA+Turkey",
            "Ortadoğu": "MEA+Turkey",
            "Kuzey Amerika": "North America",
            "Türkiye": "Turkey",
        }
        df_long["region"] = df_long["region_label"].map(region_map).fillna(
            df_long["region_label"]
        )
        df_long["technology"] = tech_label
        df_long = df_long.rename(columns={"Yıllar": "year"})
        df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
        return df_long[["year", "region", "technology", "value"]]

    sam_pdlc_reg = reshape_regional("PDLC_Regional", "PDLC")
    sam_spd_reg = reshape_regional("SPD_Regional", "SPD")
    sam_tot_reg = reshape_regional("Total_Regional", "Total")

    # --- Teknoloji toplamları (global) ---
    raw_tech = pd.read_excel(excel_path, sheet_name="Technology_Total", header=None)
    df_tech = raw_tech.iloc[1:].copy()
    df_tech.columns = raw_tech.iloc[1].tolist()
    df_tech = df_tech[df_tech["Yıllar"] != "Yıllar"]
    df_tech["Yıllar"] = pd.to_numeric(df_tech["Yıllar"], errors="coerce")
    df_tech = df_tech[df_tech["Yıllar"].notna()]
    df_tech["Yıllar"] = df_tech["Yıllar"].astype(int)

    tech_cols = [c for c in df_tech.columns if c != "Yıllar"]
    tech_total_long = df_tech.melt(
        id_vars=["Yıllar"],
        value_vars=tech_cols,
        var_name="technology",
        value_name="value",
    ).rename(columns={"Yıllar": "year"})

    # --- Cephe bazlı SAM (global değerler) ---
    def reshape_facade_simple(sheet_name: str, tech_label: str, facade_label: str):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        year_col = df.columns[0]
        val_col = df.columns[1]

        df = df.dropna(subset=[year_col])
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[df[year_col].notna()]
        df[year_col] = df[year_col].astype(int)
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

        df = df.rename(columns={year_col: "year", val_col: "value"})
        df["technology"] = tech_label
        df["facade"] = facade_label
        return df[["year", "technology", "facade", "value"]]

    sam_pdlc_dis = reshape_facade_simple(
        "PDLC_Dis_Cephe_Yillik", "PDLC", "Dış Cephe"
    )
    sam_pdlc_ic = reshape_facade_simple(
        "PDLC_Ic_Cephe_Yillik", "PDLC", "İç Cephe"
    )
    sam_spd_dis = reshape_facade_simple(
        "SPD_Dis_Cephe_Yillik", "SPD", "Dış Cephe"
    )

    return {
        "pdlc_regional": sam_pdlc_reg,
        "spd_regional": sam_spd_reg,
        "total_regional": sam_tot_reg,
        "tech_total_long": tech_total_long,
        "pdlc_dis": sam_pdlc_dis,
        "pdlc_ic": sam_pdlc_ic,
        "spd_dis": sam_spd_dis,
    }


# ============================================================
# 4) MİMARİ – SOM VERİLERİ (SOM.xlsx)
# ============================================================

@st.cache_data
def load_som_data(excel_path: str):
    """
    SOM.xlsx:
      - 'PDLC' sheet: firmanın PDLC ile girebileceği pazarlar
      - 'SPD' sheet: 2028'e kadar hazırlık, sonrası giriş
    Yapı:
      satır0: "PDLC"/"SPD"
      satır1: Yıllar / Avrupa+Rusya / ... / Toplam
      satır2: segment açıklamaları
      satır3+: yıllar
    Dönen:
      som_pdlc: year | value
      som_spd:  year | value
      som_total: year | pdlc | spd | total
    """
    raw_pdlc = pd.read_excel(excel_path, sheet_name="PDLC", header=None)
    raw_spd = pd.read_excel(excel_path, sheet_name="SPD", header=None)

    def reshape_som_raw(raw):
        header = raw.iloc[1].tolist()
        df = raw.iloc[3:].copy()
        df.columns = header

        df = df[df["Yıllar"].notna()]
        df["Yıllar"] = pd.to_numeric(df["Yıllar"], errors="coerce")
        df = df[df["Yıllar"].notna()]
        df["Yıllar"] = df["Yıllar"].astype(int)
        df["Toplam"] = pd.to_numeric(df["Toplam"], errors="coerce")

        return df[["Yıllar", "Toplam"]].rename(
            columns={"Yıllar": "year", "Toplam": "value"}
        )

    som_pdlc = reshape_som_raw(raw_pdlc)
    som_spd = reshape_som_raw(raw_spd)

    years = sorted(set(som_pdlc["year"]).union(set(som_spd["year"])))
    df_total = pd.DataFrame({"year": years})
    df_total = df_total.merge(som_pdlc, on="year", how="left")
    df_total = df_total.rename(columns={"value": "pdlc"})
    df_total = df_total.merge(som_spd, on="year", how="left")
    df_total = df_total.rename(columns={"value": "spd"})

    df_total["pdlc"] = df_total["pdlc"].fillna(0)
    df_total["spd"] = df_total["spd"].fillna(0)
    df_total["total"] = df_total["pdlc"] + df_total["spd"]

    return som_pdlc, som_spd, df_total


# ============================================================
# 5) OTOMOTİV – TEK TABLO VERİ (Otomotiv.xlsx)
# ============================================================

# ============================================================
# 5) OTOMOTİV – TAM + SAM + SOM VERİLERİ (Otomotiv.xlsx)
# ============================================================

@st.cache_data
def load_auto_data(excel_path: str) -> pd.DataFrame:
    """
    Otomotiv.xlsx içindeki:
      - 'TAM' sheet:  Araç Adet, Cam (m2)
      - 'SAM' sheet:  PDLC (m2), SPD (m2), Toplam Akıllı Cam (m2)
      - 'SOM' sheet:  PDLC SOM (m2), SPD SOM (m2), Toplam SOM (m2)

    hepsini 'year' üzerinden birleştirip tek bir DataFrame döner.
    """

    # ---------- TAM ----------
    raw_tam = pd.read_excel(excel_path, sheet_name="TAM", header=None)
    header_tam = raw_tam.iloc[1].tolist()
    tam = raw_tam.iloc[2:, :len(header_tam)].copy()
    tam.columns = header_tam  # [NaN, 'Araç Adet', 'Cam (m2)']

    first_tam_col = tam.columns[0]
    tam = tam.rename(columns={first_tam_col: "year"})
    tam = tam.dropna(subset=["year"])
    tam["year"] = pd.to_numeric(tam["year"], errors="coerce")
    tam = tam[tam["year"].notna()]
    tam["year"] = tam["year"].astype(int)

    # ---------- SAM ----------
    raw_sam = pd.read_excel(excel_path, sheet_name="SAM", header=None)
    header_sam = raw_sam.iloc[1].tolist()
    sam = raw_sam.iloc[2:, :len(header_sam)].copy()
    sam.columns = header_sam  # [NaN, 'PDLC (m2)', 'SPD (m2)', 'Toplam Akıllı Cam (m2)']

    first_sam_col = sam.columns[0]
    sam = sam.rename(columns={first_sam_col: "year"})
    sam = sam.dropna(subset=["year"])
    sam["year"] = pd.to_numeric(sam["year"], errors="coerce")
    sam = sam[sam["year"].notna()]
    sam["year"] = sam["year"].astype(int)

    # ---------- SOM ----------
    raw_som = pd.read_excel(excel_path, sheet_name="SOM", header=None)
    header_som = raw_som.iloc[1].tolist()
    som = raw_som.iloc[2:, :len(header_som)].copy()
    som.columns = header_som  # [NaN, 'PDLC SOM (m2)', 'SPD SOM (m2)', 'Toplam SOM (m2)']

    first_som_col = som.columns[0]
    som = som.rename(columns={first_som_col: "year"})
    som = som.dropna(subset=["year"])
    som["year"] = pd.to_numeric(som["year"], errors="coerce")
    som = som[som["year"].notna()]
    som["year"] = som["year"].astype(int)

    # ---------- BİRLEŞTİR ----------
    df = tam.merge(sam, on="year", how="left").merge(som, on="year", how="left")

    # Tüm sayısal kolonları numeric yap
    numeric_cols = [c for c in df.columns if c != "year"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Kolonları mantıklı sıraya koy
    cols_order = [
        "year",
        "Araç Adet",
        "Cam (m2)",
        "PDLC (m2)",
        "SPD (m2)",
        "Toplam Akıllı Cam (m2)",
        "PDLC SOM (m2)",
        "SPD SOM (m2)",
        "Toplam SOM (m2)",
    ]
    cols_order = [c for c in cols_order if c in df.columns]

    return df[cols_order]

# ============================================================
# 6) SAYFA – MİMARİ: TAM YILLIK ANALİZ
# ============================================================

def page_tam_explorer(df_main: pd.DataFrame):
    st.header("TAM – Yıllık Analiz (İç / Dış Cephe) – Mimari")

    st.sidebar.subheader("TAM Yıllık Analiz Ayarları (Mimari)")

    unit_label = st.sidebar.selectbox("Birim", ["m²", "USD"])
    unit_key = "m2" if unit_label == "m²" else "USD"
    unit_col = get_unit_column(unit_key)

    facade_option = st.sidebar.radio(
        "Grafikte gösterilecek cephe",
        ["Her ikisi", "İç Cephe", "Dış Cephe"],
        index=0,
    )

    breakdown_facade = st.sidebar.radio(
        "Breakdown için cephe",
        ["İç Cephe", "Dış Cephe"],
        index=0,
    )

    min_year = int(df_main["year"].min())
    max_year = int(df_main["year"].max())

    selected_year = st.sidebar.slider(
        "Yıl seç",
        min_value=min_year,
        max_value=max_year,
        value=min_year,
        step=1,
    )

    def aggregate_year_totals(df, facade):
        f = df[df["facade"] == facade]
        out = (
            f.groupby("year")[unit_col]
            .sum()
            .reset_index()
            .rename(columns={unit_col: "total_value"})
            .sort_values("year")
        )
        return out

    def get_regional_breakdown(df, year, facade):
        f = df[(df["facade"] == facade) & (df["year"] == year)]
        out = (
            f.groupby("region")[unit_col]
            .sum()
            .reset_index()
            .rename(columns={unit_col: "value"})
        )
        return out

    if facade_option == "Her ikisi":
        facades_to_plot = ["İç Cephe", "Dış Cephe"]
    else:
        facades_to_plot = [facade_option]

    facade_colors = {
        "İç Cephe": "#ff7f0e",
        "Dış Cephe": "#1f77b4",
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            "Toplam TAM – Yıllara Göre",
            "Bölgesel Breakdown",
        ),
    )

    # Sol: yıllara göre TAM
    for facade in facades_to_plot:
        agg = aggregate_year_totals(df_main, facade=facade)
        sizes = [24 if y == selected_year else 12 for y in agg["year"]]

        fig.add_trace(
            go.Scatter(
                x=agg["year"],
                y=agg["total_value"],
                mode="lines+markers",
                name=facade,
                marker=dict(
                    size=sizes,
                    color=facade_colors.get(facade, None),
                    line=dict(width=1),
                ),
                hovertemplate=(
                    "Yıl: %{x}<br>"
                    "TAM (" + unit_label + "): %{y:,.0f}<br>"
                    f"Cephe: {facade}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    breakdown_df = get_regional_breakdown(
        df_main, year=selected_year, facade=breakdown_facade
    )

    fig.add_trace(
        go.Bar(
            x=breakdown_df["region"],
            y=breakdown_df["value"],
            text=[f"{v:,.0f}" for v in breakdown_df["value"]],
            textposition="outside",
            name=f"{breakdown_facade} – {selected_year}",
            hovertemplate="Bölge: %{x}<br>Değer: %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    y_axis_title = f"TAM ({unit_label})"
    fig.update_xaxes(title_text="Yıl", row=1, col=1)
    fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
    fig.update_yaxes(title_text=y_axis_title, row=1, col=2)

    fig.update_layout(
        height=550,
        margin=dict(l=60, r=40, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    if len(fig.layout.annotations) > 1:
        fig.layout.annotations[1].text = (
            f"Bölgesel Breakdown – {breakdown_facade} – {selected_year}"
        )

    st.plotly_chart(fig, use_container_width=True)

    total_selected = breakdown_df["value"].sum()
    st.markdown(
        f"""
        **Seçili yıl:** {selected_year}  
        **Breakdown cephesi:** {breakdown_facade}  
        **Toplam TAM ({unit_label}):** {total_selected:,.0f}
        """
    )


# ============================================================
# 7) SAYFA – MİMARİ: TAM SEGMENT DASHBOARD
# ============================================================

def page_tam_segment_dashboard(df_dis_long: pd.DataFrame, df_ic_long: pd.DataFrame):
    st.header("TAM Segment Dashboard – Dış / İç Cephe – Mimari")

    dis_seg = df_dis_long.groupby("segment")["value"].sum().reset_index()
    ic_seg = df_ic_long.groupby("segment")["value"].sum().reset_index()

    pivot_dis = df_dis_long.pivot_table(
        index="segment", columns="region", values="value", aggfunc="sum"
    )
    pivot_ic = df_ic_long.pivot_table(
        index="segment", columns="region", values="value", aggfunc="sum"
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "domain"}, {"type": "domain"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        subplot_titles=[
            "Dış Cephe – Segmentler",
            "İç Cephe – Segmentler",
            "Dış Cephe – Segment x Bölge",
            "İç Cephe – Segment x Bölge",
        ],
    )

    fig.add_trace(
        go.Pie(
            labels=dis_seg["segment"],
            values=dis_seg["value"],
            hole=0.35,
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="auto",
            hovertemplate="%{label}<br>%{value:,.0f} m²<br>%{percent}<extra></extra>",
            name="Dış Cephe",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=ic_seg["segment"],
            values=ic_seg["value"],
            hole=0.35,
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="auto",
            hovertemplate="%{label}<br>%{value:,.0f} m²<br>%{percent}<extra></extra>",
            name="İç Cephe",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=pivot_dis.values,
            x=pivot_dis.columns,
            y=pivot_dis.index,
            colorscale="Blues",
            text=pivot_dis.values,
            texttemplate="%{text:.0f}",
            colorbar=dict(title="m²"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=pivot_ic.values,
            x=pivot_ic.columns,
            y=pivot_ic.index,
            colorscale="Blues",
            text=pivot_ic.values,
            texttemplate="%{text:.0f}",
            showscale=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=950,
        margin=dict(t=90, b=60, l=60, r=60),
        title_text="Smart Glass TAM Segment Dashboard – Dış / İç Cephe",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
        ),
    )

    for ann in fig.layout.annotations:
        ann.font.size = 12

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 8) SAYFA – MİMARİ: SAM OVERVIEW
# ============================================================

def page_sam_overview(sam_data: dict):
    st.header("SAM Overview – PDLC / SPD / Toplam – Mimari")

    tech_total_long = sam_data["tech_total_long"]
    sam_pdlc_reg = sam_data["pdlc_regional"]
    sam_spd_reg = sam_data["spd_regional"]
    sam_pdlc_dis = sam_data["pdlc_dis"]
    sam_pdlc_ic = sam_data["pdlc_ic"]
    sam_spd_dis = sam_data["spd_dis"]

    # --- A) Teknoloji SAM Gelişimi (global) ---
    st.subheader("Teknoloji Bazlı SAM Gelişimi (Global)")

    fig1 = go.Figure()
    for tech in tech_total_long["technology"].unique():
        df_t = tech_total_long[tech_total_long["technology"] == tech].sort_values(
            "year"
        )
        fig1.add_trace(
            go.Scatter(
                x=df_t["year"],
                y=df_t["value"],
                mode="lines+markers",
                name=tech,
                hovertemplate="Yıl: %{x}<br>SAM: %{y:,.0f} m²<extra></extra>",
            )
        )

    fig1.update_layout(
        height=400,
        xaxis_title="Yıl",
        yaxis_title="SAM (m²)",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- B) Bölgesel SAM – PDLC / SPD ---
    st.subheader("Bölgesel SAM – PDLC vs SPD")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        df = sam_pdlc_reg
        for region in df["region"].unique():
            d = df[df["region"] == region].sort_values("year")
            fig2.add_trace(
                go.Scatter(
                    x=d["year"],
                    y=d["value"],
                    mode="lines+markers",
                    name=region,
                    hovertemplate="Yıl: %{x}<br>PDLC SAM: %{y:,.0f} m²<extra></extra>",
                )
            )
        fig2.update_layout(
            title="PDLC – Bölgesel SAM",
            xaxis_title="Yıl",
            yaxis_title="SAM (m²)",
            height=400,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        df = sam_spd_reg
        for region in df["region"].unique():
            d = df[df["region"] == region].sort_values("year")
            fig3.add_trace(
                go.Scatter(
                    x=d["year"],
                    y=d["value"],
                    mode="lines+markers",
                    name=region,
                    hovertemplate="Yıl: %{x}<br>SPD SAM: %{y:,.0f} m²<extra></extra>",
                )
            )
        fig3.update_layout(
            title="SPD – Bölgesel SAM",
            xaxis_title="Yıl",
            yaxis_title="SAM (m²)",
            height=400,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- C) Cephe Bazlı SAM – PDLC / SPD ---
    st.subheader("Cephe Bazlı SAM – PDLC / SPD (Global)")

    fig4 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Dış Cephe", "İç Cephe"],
        column_widths=[0.5, 0.5],
    )

    df_dis = pd.concat([sam_pdlc_dis, sam_spd_dis], ignore_index=True)
    for tech in df_dis["technology"].unique():
        d = (
            df_dis[df_dis["technology"] == tech]
            .groupby("year")["value"]
            .sum()
            .reset_index()
        )
        fig4.add_trace(
            go.Scatter(
                x=d["year"],
                y=d["value"],
                mode="lines+markers",
                name=f"{tech} – Dış Cephe",
                hovertemplate="Yıl: %{x}<br>SAM: %{y:,.0f} m²<extra></extra>",
            ),
            row=1,
            col=1,
        )

    d_ic = sam_pdlc_ic.groupby("year")["value"].sum().reset_index()
    fig4.add_trace(
        go.Scatter(
            x=d_ic["year"],
            y=d_ic["value"],
            mode="lines+markers",
            name="PDLC – İç Cephe",
            hovertemplate="Yıl: %{x}<br>SAM: %{y:,.0f} m²<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig4.update_xaxes(title_text="Yıl", row=1, col=1)
    fig4.update_xaxes(title_text="Yıl", row=1, col=2)
    fig4.update_yaxes(title_text="SAM (m²)", row=1, col=1)
    fig4.update_yaxes(title_text="SAM (m²)", row=1, col=2)

    fig4.update_layout(
        height=450,
        margin=dict(l=60, r=40, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# 9) SAYFA – MİMARİ: SOM STRATEGY
# ============================================================

def page_som_strategy(df_tam: pd.DataFrame, sam_data: dict, som_pdlc, som_spd, som_total):
    st.header("SOM Strategy Dashboard – PDLC / SPD – Mimari")

    # --- A) SOM – PDLC, SPD, Toplam ---
    st.subheader("Teknoloji Bazlı SOM Gelişimi (Global)")

    fig1 = go.Figure()
    d = som_total.sort_values("year")

    fig1.add_trace(
        go.Scatter(
            x=d["year"],
            y=d["pdlc"],
            mode="lines+markers",
            name="PDLC SOM",
            hovertemplate="Yıl: %{x}<br>PDLC SOM: %{y:,.0f} m²<extra></extra>",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=d["year"],
            y=d["spd"],
            mode="lines+markers",
            name="SPD SOM",
            hovertemplate="Yıl: %{x}<br>SPD SOM: %{y:,.0f} m²<extra></extra>",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=d["year"],
            y=d["total"],
            mode="lines+markers",
            name="Toplam SOM",
            hovertemplate="Yıl: %{x}<br>Toplam SOM: %{y:,.0f} m²<extra></extra>",
        )
    )

    fig1.update_layout(
        height=400,
        xaxis_title="Yıl",
        yaxis_title="SOM (m²)",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- B) TAM / SAM / SOM karşılaştırma + penetrasyon ---
    st.subheader("TAM / SAM / SOM Karşılaştırması ve Penetrasyon")

    tech_total_long = sam_data["tech_total_long"]

    sam_total_by_year = (
        tech_total_long.groupby("year")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "sam_total"})
    )

    tam_total_by_year = (
        df_tam.groupby("year")["value_m2"]
        .sum()
        .reset_index()
        .rename(columns={"value_m2": "tam_total"})
    )

    som_tot = som_total[["year", "total"]].rename(columns={"total": "som_total"})

    df_comb = pd.DataFrame(
        {
            "year": sorted(
                set(tam_total_by_year["year"])
                | set(sam_total_by_year["year"])
                | set(som_tot["year"])
            )
        }
    )
    df_comb = df_comb.merge(tam_total_by_year, on="year", how="left")
    df_comb = df_comb.merge(sam_total_by_year, on="year", how="left")
    df_comb = df_comb.merge(som_tot, on="year", how="left")

    df_comb[["tam_total", "sam_total", "som_total"]] = df_comb[
        ["tam_total", "sam_total", "som_total"]
    ].fillna(0)

    df_comb["som_over_tam"] = df_comb["som_total"] / df_comb["tam_total"].replace(
        0, pd.NA
    )
    df_comb["som_over_sam"] = df_comb["som_total"] / df_comb["sam_total"].replace(
        0, pd.NA
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=df_comb["year"],
            y=df_comb["tam_total"],
            mode="lines+markers",
            name="TAM",
            hovertemplate="Yıl: %{x}<br>TAM: %{y:,.0f} m²<extra></extra>",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=df_comb["year"],
            y=df_comb["sam_total"],
            mode="lines+markers",
            name="SAM",
            hovertemplate="Yıl: %{x}<br>SAM: %{y:,.0f} m²<extra></extra>",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=df_comb["year"],
            y=df_comb["som_total"],
            mode="lines+markers",
            name="SOM",
            hovertemplate="Yıl: %{x}<br>SOM: %{y:,.0f} m²<extra></extra>",
        )
    )

    fig2.update_layout(
        height=400,
        xaxis_title="Yıl",
        yaxis_title="Hacim (m²)",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SOM Penetrasyon Oranları")

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=df_comb["year"],
            y=df_comb["som_over_tam"] * 100,
            mode="lines+markers",
            name="SOM / TAM",
            hovertemplate="Yıl: %{x}<br>%{y:.1f} %<extra></extra>",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=df_comb["year"],
            y=df_comb["som_over_sam"] * 100,
            mode="lines+markers",
            name="SOM / SAM",
            hovertemplate="Yıl: %{x}<br>%{y:.1f} %<extra></extra>",
        )
    )

    fig3.update_layout(
        height=350,
        xaxis_title="Yıl",
        yaxis_title="Penetrasyon (%)",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# 10) SAYFA – OTOMOTİV: TAM
# ============================================================

def page_auto_tam(df_auto: pd.DataFrame):
    st.header("TAM – Yıllık Analiz – Otomotiv")

    st.sidebar.subheader("TAM Ayarları (Otomotiv)")

    min_year = int(df_auto["year"].min())
    max_year = int(df_auto["year"].max())
    selected_year = st.sidebar.slider(
        "Yıl (bilgi amaçlı)", min_year, max_year, min_year, step=1
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_auto["year"],
            y=df_auto["Araç Adet"],
            name="Araç Adedi",
            hovertemplate="Yıl: %{x}<br>Araç: %{y:,.0f} adet<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df_auto["year"],
            y=df_auto["Cam (m2)"],
            mode="lines+markers",
            name="Toplam Cam (m²)",
            hovertemplate="Yıl: %{x}<br>Cam: %{y:,.0f} m²<extra></extra>",
        ),
        secondary_y=False,
    )

    if "Toplam Akıllı Cam (m2)" in df_auto.columns:
        fig.add_trace(
            go.Scatter(
                x=df_auto["year"],
                y=df_auto["Toplam Akıllı Cam (m2)"],
                mode="lines+markers",
                name="Akıllı Cam Potansiyeli (m²)",
                hovertemplate="Yıl: %{x}<br>Akıllı Cam Pot.: %{y:,.0f} m²<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_xaxes(title_text="Yıl")
    fig.update_yaxes(title_text="Cam Alanı (m²)", secondary_y=False)
    fig.update_yaxes(title_text="Araç Adedi", secondary_y=True)

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        title="Otomotiv – TAM: Araç Adedi ve Cam Alanı",
    )

    st.plotly_chart(fig, use_container_width=True)

    row_sel = df_auto[df_auto["year"] == selected_year].iloc[0]
    st.markdown(
        f"""
        **Seçili yıl: {selected_year}**

        - Araç adedi: **{row_sel['Araç Adet']:,.0f}**
        - Toplam cam alanı: **{row_sel['Cam (m2)']:,.0f} m²**
        - Toplam akıllı cam potansiyeli: **{row_sel.get('Toplam Akıllı Cam (m2)', 0):,.0f} m²**
        """
    )


# ============================================================
# 11) SAYFA – OTOMOTİV: SAM
# ============================================================

def page_auto_sam_overview(df_auto: pd.DataFrame):
    st.header("SAM Overview – Otomotiv (PDLC / SPD / Toplam Akıllı Cam)")

    # Referans olsun diye kolonları görelim (ilk çalıştırmada çok yardımcı olur;
    # her şey yoluna girdikten sonra bu satırı istersen silebilirsin)
    st.caption("Otomotiv verisindeki kolonlar:")
    st.write(list(df_auto.columns))

    # Bizim bildiğimiz net kolon isimleri (Otomotiv.xlsx'ten kontrol ettik):
    # 'PDLC (m2)', 'SPD (m2)', 'Toplam Akıllı Cam (m2)'
    fig = go.Figure()

    if "PDLC (m2)" in df_auto.columns:
        fig.add_trace(
            go.Scatter(
                x=df_auto["year"],
                y=df_auto["PDLC (m2)"],
                mode="lines+markers",
                name="PDLC SAM (m²)",
                hovertemplate="Yıl: %{x}<br>PDLC SAM: %{y:,.0f} m²<extra></extra>",
            )
        )

    if "SPD (m2)" in df_auto.columns:
        fig.add_trace(
            go.Scatter(
                x=df_auto["year"],
                y=df_auto["SPD (m2)"],
                mode="lines+markers",
                name="SPD SAM (m²)",
                hovertemplate="Yıl: %{x}<br>SPD SAM: %{y:,.0f} m²<extra></extra>",
            )
        )

    if "Toplam Akıllı Cam (m2)" in df_auto.columns:
        fig.add_trace(
            go.Scatter(
                x=df_auto["year"],
                y=df_auto["Toplam Akıllı Cam (m2)"],
                mode="lines+markers",
                name="Toplam Akıllı Cam SAM (m²)",
                hovertemplate="Yıl: %{x}<br>Toplam Akıllı Cam SAM: %{y:,.0f} m²<extra></extra>",
            )
        )

    fig.update_layout(
        height=450,
        xaxis_title="Yıl",
        yaxis_title="SAM (m²)",
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        title="Otomotiv – SAM Gelişimi",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Küçük bir tablo da gösterelim
    cols_to_show = ["year"]
    for c in ["PDLC (m2)", "SPD (m2)", "Toplam Akıllı Cam (m2)"]:
        if c in df_auto.columns:
            cols_to_show.append(c)

    st.subheader("Veri Tablosu (SAM)")
    st.dataframe(df_auto[cols_to_show])

# ============================================================
# 12) SAYFA – OTOMOTİV: SOM
# ============================================================

def page_auto_som_strategy(df_auto: pd.DataFrame):
    st.header("SOM Strategy – Otomotiv (PDLC / SPD / Toplam SOM)")

    df = df_auto.copy()

    # Kolonları bire bir isimleriyle kullanacağız:
    col_pdlc_som = "PDLC SOM (m2)" if "PDLC SOM (m2)" in df.columns else None
    col_spd_som  = "SPD SOM (m2)" if "SPD SOM (m2)" in df.columns else None
    col_total_som = "Toplam SOM (m2)" if "Toplam SOM (m2)" in df.columns else None

    # Eğer Toplam SOM yoksa kendimiz oluşturalım
    if col_total_som is None:
        if col_pdlc_som or col_spd_som:
            df["Toplam SOM (m2)"] = 0
            if col_pdlc_som:
                df["Toplam SOM (m2)"] += df[col_pdlc_som]
            if col_spd_som:
                df["Toplam SOM (m2)"] += df[col_spd_som]
            col_total_som = "Toplam SOM (m2)"

    # A) SOM çizgileri
    fig1 = go.Figure()

    if col_pdlc_som:
        fig1.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[col_pdlc_som],
                mode="lines+markers",
                name="PDLC SOM",
                hovertemplate="Yıl: %{x}<br>PDLC SOM: %{y:,.0f} m²<extra></extra>",
            )
        )

    if col_spd_som:
        fig1.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[col_spd_som],
                mode="lines+markers",
                name="SPD SOM",
                hovertemplate="Yıl: %{x}<br>SPD SOM: %{y:,.0f} m²<extra></extra>",
            )
        )

    if col_total_som:
        fig1.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[col_total_som],
                mode="lines+markers",
                name="Toplam SOM",
                hovertemplate="Yıl: %{x}<br>Toplam SOM: %{y:,.0f} m²<extra></extra>",
            )
        )

    fig1.update_layout(
        height=400,
        xaxis_title="Yıl",
        yaxis_title="SOM (m²)",
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        title="Otomotiv – SOM Gelişimi",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # B) Penetrasyon (SOM/TAM, SOM/SAM)
    st.subheader("SOM Penetrasyon Oranları – Otomotiv")

    if col_total_som and "Cam (m2)" in df.columns and "Toplam Akıllı Cam (m2)" in df.columns:
        df_pen = df.copy()

        df_pen["som_over_tam"] = df_pen[col_total_som] / df_pen["Cam (m2)"].replace(0, pd.NA)
        df_pen["som_over_sam"] = df_pen[col_total_som] / df_pen["Toplam Akıllı Cam (m2)"].replace(0, pd.NA)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df_pen["year"],
                y=df_pen["som_over_tam"] * 100,
                mode="lines+markers",
                name="SOM / TAM",
                hovertemplate="Yıl: %{x}<br>%{y:.1f} %<extra></extra>",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=df_pen["year"],
                y=df_pen["som_over_sam"] * 100,
                mode="lines+markers",
                name="SOM / SAM",
                hovertemplate="Yıl: %{x}<br>%{y:.1f} %<extra></extra>",
            )
        )

        fig2.update_layout(
            height=350,
            xaxis_title="Yıl",
            yaxis_title="Penetrasyon (%)",
            margin=dict(l=60, r=40, t=40, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            title="Otomotiv – SOM Penetrasyon Oranları",
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("SOM penetrasyonunu hesaplamak için 'Cam (m2)', 'Toplam Akıllı Cam (m2)' ve SOM kolonları gerekli.")

# ============================================================
# 13) MAIN – STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Smart Glass TAM–SAM–SOM Explorer",
        layout="wide",
    )

    st.sidebar.title("Smart Glass TAM–SAM–SOM Explorer")

    segment_type = st.sidebar.radio(
        "Segment tipi",
        ["Mimari", "Otomotiv"],
        index=0,
    )

    page = st.sidebar.radio(
        "Sayfa Seç",
        [
            "TAM – Yıllık Analiz",
            "TAM – Segment Dashboard",
            "SAM – Overview",
            "SOM – Strategy Dashboard",
        ],
        index=0,
    )

    if segment_type == "Mimari":
        tam_path = "TAM.xlsx"
        sam_path = "SAM.xlsx"
        som_path = "SOM.xlsx"

        df_tam_main = load_tam_yearly(tam_path)
        df_dis_long, df_ic_long = load_tam_segment_distributions(tam_path)
        sam_data = load_sam_data(sam_path)
        som_pdlc, som_spd, som_total = load_som_data(som_path)

        if page == "TAM – Yıllık Analiz":
            page_tam_explorer(df_tam_main)
        elif page == "TAM – Segment Dashboard":
            page_tam_segment_dashboard(df_dis_long, df_ic_long)
        elif page == "SAM – Overview":
            page_sam_overview(sam_data)
        elif page == "SOM – Strategy Dashboard":
            page_som_strategy(df_tam_main, sam_data, som_pdlc, som_spd, som_total)

    else:  # Otomotiv
        auto_path = "Otomotiv.xlsx"
        df_auto = load_auto_data(auto_path)

        if page == "TAM – Yıllık Analiz":
            page_auto_tam(df_auto)
        elif page == "TAM – Segment Dashboard":
            st.header("TAM Segment Dashboard – Otomotiv")
            st.info("Otomotiv için segment bazlı (sunroof / yan cam / vb.) kırılım henüz modellenmedi.")
        elif page == "SAM – Overview":
            page_auto_sam_overview(df_auto)
        elif page == "SOM – Strategy Dashboard":
            page_auto_som_strategy(df_auto)


if __name__ == "__main__":
    main()
