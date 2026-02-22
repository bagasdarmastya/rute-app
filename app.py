import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import datetime
import io
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Audit Route Planner", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'df_main' not in st.session_state:
    st.session_state.df_main = None

# --- FUNGSI BANTUAN ---

@st.cache_data(show_spinner=False) 
def get_coordinates(df):
    """Mengubah alamat menjadi koordinat (Geocoding)"""
    geolocator = Nominatim(user_agent="audit_planner_app_v7_1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
    
    def make_full_address(row):
        parts = [
            str(row.get('ALAMAT', '')),
            str(row.get('KELURAHAN', '')),
            str(row.get('KECAMATAN', '')),
            str(row.get('KABUPATEN', ''))
        ]
        clean_parts = [p for p in parts if p and p.lower() != 'nan' and p.lower() != 'none']
        return ", ".join(clean_parts)

    df_result = df.copy()
    df_result['Full_Address_Search'] = df_result.apply(make_full_address, axis=1)
    
    total_rows = len(df_result)
    results = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, address in enumerate(df_result['Full_Address_Search']):
        if i % 5 == 0:
            status_text.text(f"Mencari koordinat {i+1}/{total_rows}...")
            progress_bar.progress((i + 1) / total_rows)
            
        try:
            if not address.strip(): 
                results.append((None, None))
                continue
            location = geocode(address)
            if location:
                results.append((location.latitude, location.longitude))
            else:
                results.append((None, None))
        except Exception:
            results.append((None, None)) 
        
    progress_bar.empty()
    status_text.empty()
    
    if results:
        df_result['Latitude'], df_result['Longitude'] = zip(*results)
    else:
        df_result['Latitude'] = None
        df_result['Longitude'] = None
        
    return df_result

def optimize_route_initial(df, num_routes):
    """Langkah 1: Clustering Awal berdasarkan Geografis"""
    df_clean = df.dropna(subset=['Latitude', 'Longitude']).copy()
    if len(df_clean) == 0: return None
    if len(df_clean) < num_routes: num_routes = len(df_clean)

    X = df_clean[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=num_routes, random_state=42, n_init='auto')
    df_clean['Route_ID'] = kmeans.fit_predict(X)
    df_clean['Route_ID'] = df_clean['Route_ID'] + 1
    return df_clean

def calculate_route_sequence(df_route, start_time, end_time_limit, duration_per_visit, office_coords):
    """
    Menghitung urutan kunjungan dalam satu rute dengan batasan waktu yang ketat.
    """
    visits = df_route.copy()
    current_loc = office_coords
    current_time = datetime.datetime.combine(datetime.date.today(), start_time)
    limit_dt = datetime.datetime.combine(datetime.date.today(), end_time_limit)
    
    ordered_indices = []
    overflow_indices = []
    
    while len(visits) > 0:
        # Cari toko terdekat dari posisi terakhir
        visits['dist_temp'] = visits.apply(
            lambda row: geodesic(current_loc, (row['Latitude'], row['Longitude'])).km, axis=1
        )
        nearest_idx = visits['dist_temp'].idxmin()
        nearest = visits.loc[nearest_idx]
        
        # Hitung Waktu
        travel_hours = nearest['dist_temp'] / 30 # 30 km/jam
        arrival_time = current_time + datetime.timedelta(hours=travel_hours)
        finish_time = arrival_time + datetime.timedelta(minutes=duration_per_visit)
        
        # Hitung perjalanan pulang dari toko ini ke kantor
        dist_to_home = geodesic((nearest['Latitude'], nearest['Longitude']), office_coords).km
        time_to_home = dist_to_home / 30 
        arrival_at_office = finish_time + datetime.timedelta(hours=time_to_home)
        
        # LOGIKA BARU: Jika pulang telat, JANGAN dimasukkan ke rute (masuk overflow)
        if arrival_at_office <= limit_dt:
            df_route.at[nearest_idx, 'Jam_Datang'] = arrival_time.strftime("%H:%M")
            df_route.at[nearest_idx, 'Jam_Selesai'] = finish_time.strftime("%H:%M")
            df_route.at[nearest_idx, 'Jarak_KM'] = round(nearest['dist_temp'], 2)
            
            ordered_indices.append(nearest_idx)
            current_loc = (nearest['Latitude'], nearest['Longitude'])
            current_time = finish_time
        else:
            overflow_indices.append(nearest_idx)
        
        visits = visits.drop(nearest_idx)
    
    # Buat DataFrame Valid
    valid_df = df_route.loc[ordered_indices].copy()
    valid_df['Urutan_Kunjungan'] = range(1, len(valid_df) + 1)
    
    # Tambahkan baris Pulang ke Kantor
    if not valid_df.empty:
        last_stop = valid_df.iloc[-1]
        dist_home = geodesic((last_stop['Latitude'], last_stop['Longitude']), office_coords).km
        time_home = dist_home / 30
        last_finish = datetime.datetime.strptime(last_stop['Jam_Selesai'], "%H:%M").time()
        start_home_dt = datetime.datetime.combine(datetime.date.today(), last_finish)
        arrive_home_dt = start_home_dt + datetime.timedelta(hours=time_home)
        
        return_row = pd.DataFrame([{
            'Route_ID': valid_df.iloc[0]['Route_ID'],
            'Urutan_Kunjungan': len(valid_df) + 1,
            'CUSTOMER NAME': '🏢 KEMBALI KE KANTOR',
            'ALAMAT': 'Selesai',
            'Jam_Datang': arrive_home_dt.strftime("%H:%M"),
            'Jam_Selesai': '-',
            'Jarak_KM': round(dist_home, 2),
            'Latitude': office_coords[0],
            'Longitude': office_coords[1]
        }])
        valid_df = pd.concat([valid_df, return_row], ignore_index=True)

    overflow_df = df_route.loc[overflow_indices].copy()
    return valid_df, overflow_df

def solve_routing_with_balancing(df, num_routes, start_time, end_time, duration, office_coords):
    """
    Fungsi Utama: Clustering + Sequencing + Rebalancing
    """
    # 1. Clustering Awal
    df_clustered = optimize_route_initial(df, num_routes)
    if df_clustered is None: return None, None

    final_routes = {}
    all_overflows = pd.DataFrame()
    
    # 2. Hitung Rute Awal
    for route_id in range(1, num_routes + 1):
        route_data = df_clustered[df_clustered['Route_ID'] == route_id].copy()
        valid, overflow = calculate_route_sequence(route_data, start_time, end_time, duration, office_coords)
        
        final_routes[route_id] = valid
        all_overflows = pd.concat([all_overflows, overflow])
    
    # 3. Rebalancing (Coba masukkan Overflow ke Rute Lain)
    if not all_overflows.empty:
        all_overflows = all_overflows.reset_index(drop=True)
        still_overflow = []

        for _, row in all_overflows.iterrows():
            inserted = False
            # Coba masukkan ke setiap rute yang ada
            for r_id, r_df in final_routes.items():
                real_stops = r_df[r_df['CUSTOMER NAME'] != '🏢 KEMBALI KE KANTOR'].copy()
                
                # Tambahkan titik ini ke rute kandidat
                temp_route = pd.concat([real_stops, pd.DataFrame([row])], ignore_index=True)
                # Hitung ulang validitas waktu
                v_test, o_test = calculate_route_sequence(temp_route, start_time, end_time, duration, office_coords)
                
                # Jika BERHASIL (tidak ada overflow baru), simpan rute baru ini
                if o_test.empty:
                    final_routes[r_id] = v_test
                    inserted = True
                    break 
            
            if not inserted:
                still_overflow.append(row)
        
        if still_overflow:
            all_overflows = pd.DataFrame(still_overflow)
        else:
            all_overflows = pd.DataFrame() # Kosong jika semua berhasil diselipkan

    return final_routes, all_overflows

# --- UI APLIKASI ---

st.title("auditMaps 🗺️")
st.caption("Tools Perencana Rute Audit Internal Otomatis (v7.1 Final Rebalance)")

# 1. SIDEBAR
with st.sidebar:
    st.header("1. Pengaturan Operasional")
    num_routes = st.number_input("Jumlah Tim / Mobil", min_value=1, max_value=20, value=2)
    
    st.header("2. Batasan Waktu")
    start_time = st.time_input("Jam Berangkat", value=datetime.time(8, 0))
    end_time = st.time_input("Jam Pulang (Hard Limit)", value=datetime.time(17, 0), help="Sistem akan memecah rute jika estimasi pulang melebihi jam ini.")
    visit_duration = st.number_input("Lama Audit per Toko (Menit)", min_value=10, value=45, step=5)
    
    st.header("3. Titik Awal (Kantor)")
    office_lat = st.number_input("Latitude Kantor", value=-7.2575, format="%.6f") 
    office_long = st.number_input("Longitude Kantor", value=112.7521, format="%.6f")
    
    st.divider()
    if st.button("Reset Aplikasi (Clear Cache)"):
        st.session_state.df_main = None
        st.cache_data.clear()
        st.rerun()

# 2. UPLOAD
st.write("---")
uploaded_file = st.file_uploader("Upload File Excel/CSV Data Customer", type=['xlsx', 'csv'])

if uploaded_file:
    if st.session_state.df_main is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = [str(x).upper().strip() for x in df.columns]
            
            lat_cols = ['LATITUDE', 'LAT', 'LATTITUDE', 'GARIS LINTANG']
            long_cols = ['LONGITUDE', 'LONG', 'LNG', 'GARIS BUJUR']
            rename_dict = {}
            for col in df.columns:
                if col in lat_cols: rename_dict[col] = 'Latitude'
                if col in long_cols: rename_dict[col] = 'Longitude'
            if rename_dict: df = df.rename(columns=rename_dict)
            
            st.session_state.df_main = df
        except Exception as e:
            st.error(f"Gagal: {e}")
            st.stop()
    
    df = st.session_state.df_main
    with st.expander("Lihat Data Awal", expanded=False): st.dataframe(df.head())
    
    # Validasi Kolom
    required_cols = ['CUSTOMER NAME', 'ALAMAT']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib hilang: {', '.join(missing)}")
        st.stop()

    # Cek Koordinat
    has_coords = 'Latitude' in df.columns and 'Longitude' in df.columns
    valid_coords_count = 0
    if has_coords:
        try:
            temp_lat = pd.to_numeric(df['Latitude'].astype(str).str.replace(',', '.'), errors='coerce')
            valid_coords_count = temp_lat.dropna().shape[0]
        except: valid_coords_count = 0
    
    if has_coords and valid_coords_count > 0:
        st.success(f"✅ {valid_coords_count} data memiliki koordinat valid.")
    else:
        st.warning("⚠️ Data belum memiliki koordinat GPS.")
        col1, col2 = st.columns([3, 1])
        with col1: st.info("Sistem akan mencari koordinat otomatis via Alamat.")
        with col2:
            if st.button("🔍 Cari Koordinat", type="primary"):
                with st.spinner("Geocoding..."):
                    df = get_coordinates(df)
                    st.session_state.df_main = df 
                    st.rerun()

    # GENERATE RUTE
    if 'Latitude' in df.columns and not df['Latitude'].isna().all():
        st.write("---")
        
        # Tombol download master data
        buffer_master = io.BytesIO()
        with pd.ExcelWriter(buffer_master, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Master Data')
        st.download_button("💾 Simpan Master Data (Dengan Koordinat)", buffer_master.getvalue(), "Master_Data_Audit.xlsx", "application/vnd.ms-excel")
        st.write("") 

        if st.button("🚀 Generate Rute & Jadwal (Auto Balance)", type="primary"):
            
            # Inisialisasi variabel agar tidak error 'UnboundLocalError'
            final_routes_dict = None
            overflow_df = None

            with st.spinner("1/3 Clustering Wilayah & Rebalancing..."):
                try:
                    # --- FIX: PEMBERSIHAN DATA KOORDINAT ---
                    df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.')
                    df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.')
                    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                    
                    if df.dropna(subset=['Latitude', 'Longitude']).empty:
                        st.error("Error: Kolom Latitude/Longitude tidak berisi angka yang valid.")
                        st.stop()
                        
                    # Proses Utama (Clustering + Sequencing + Rebalancing)
                    final_routes_dict, overflow_df = solve_routing_with_balancing(
                        df, num_routes, start_time, end_time, visit_duration, (office_lat, office_long)
                    )
                
                # --- PERBAIKAN: Menambahkan blok except yang hilang ---
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses rute: {e}")
                    st.stop() # Hentikan eksekusi jika error

            if final_routes_dict:
                with st.spinner("2/3 Membuat Peta Visualisasi..."):
                    valid_lats = df.dropna(subset=['Latitude'])['Latitude']
                    m = folium.Map(location=[valid_lats.mean(), df['Longitude'].mean()], zoom_start=12)
                    
                    folium.Marker([office_lat, office_long], popup="KANTOR", icon=folium.Icon(color="black", icon="building", prefix='fa')).add_to(m)
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'pink']
                    
                    for r_id, r_df in final_routes_dict.items():
                        if r_df.empty: continue
                        points = []
                        color = colors[(r_id-1) % len(colors)]
                        
                        for _, row in r_df.iterrows():
                            lat, lng = row['Latitude'], row['Longitude']
                            points.append([lat, lng])
                            
                            icon_type = "flag-checkered" if row['CUSTOMER NAME'] == '🏢 KEMBALI KE KANTOR' else "user"
                            icon_color = "black" if icon_type == "flag-checkered" else color
                            
                            popup_html = f"""
                            <b>Rute {r_id} #{row['Urutan_Kunjungan']}</b><br>
                            {row.get('CUSTOMER NAME', '-')}<br>
                            Jam: {row['Jam_Datang']}
                            """
                            folium.Marker([lat, lng], popup=popup_html, icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')).add_to(m)
                            
                        folium.PolyLine(points, color=color, weight=3, opacity=0.8, tooltip=f"Rute {r_id}").add_to(m)

                    if not overflow_df.empty:
                        for _, row in overflow_df.iterrows():
                            folium.Marker(
                                [row['Latitude'], row['Longitude']], 
                                popup=f"TIDAK TERCOVER: {row['CUSTOMER NAME']}",
                                icon=folium.Icon(color="gray", icon="ban", prefix='fa')
                            ).add_to(m)

                st.session_state.final_routes = final_routes_dict
                st.session_state.overflow_df = overflow_df
                st.session_state.map_html = m._repr_html_()
        
        # TAMPILKAN HASIL
        if 'final_routes' in st.session_state:
            routes = st.session_state.final_routes
            overflow = st.session_state.overflow_df
            
            # Info Statistik
            cols = st.columns(len(routes) + 1)
            i = 0
            for r_id, r_df in routes.items():
                visit_count = len(r_df[r_df['CUSTOMER NAME'] != '🏢 KEMBALI KE KANTOR'])
                if not r_df.empty:
                    last_time = r_df.iloc[-1]['Jam_Datang']
                else:
                    last_time = "-"
                
                if i < len(cols):
                    cols[i].metric(f"Rute {r_id}", f"{visit_count} Toko", f"Plg: {last_time}")
                i += 1
            
            if not overflow.empty:
                cols[-1].metric("Tidak Tercover", f"{len(overflow)} Toko", "Overtime", delta_color="inverse")
                st.error(f"⚠️ Ada {len(overflow)} toko yang tidak bisa dikunjungi hari ini karena batasan jam pulang.")
            else:
                st.success("✅ Semua toko berhasil masuk jadwal!")

            tab1, tab2 = st.tabs(["🗺️ Peta", "📥 Download Excel"])
            
            with tab1:
                if 'map_html' in st.session_state:
                    import streamlit.components.v1 as components
                    components.html(st.session_state.map_html, height=500)
            
            with tab2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # 1. Sheet Ringkasan
                    summary_data = []
                    for r_id, r_df in routes.items():
                        visit_count = len(r_df[r_df['CUSTOMER NAME'] != '🏢 KEMBALI KE KANTOR'])
                        if not r_df.empty:
                            last_time = r_df.iloc[-1]['Jam_Datang']
                        else:
                            last_time = "-"
                        summary_data.append({'Rute': f"RUTE {r_id}", 'Jumlah Toko': visit_count, 'Jam Sampai Kantor': last_time})
                    
                    pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Ringkasan')

                    # 2. Sheet Per Rute (Multi-Sheet Logic)
                    for r_id, r_df in routes.items():
                        display_cols = [
                            'Urutan_Kunjungan', 'Jam_Datang', 'Jam_Selesai', 
                            'CUSTOMER NAME', 'ALAMAT', 
                            'CUSTOMER CODE', 'NO. DOKUMEN', 'NOMINAL', 'KETERANGAN',
                            'Jarak_KM' # Kolom ini mungkin NaN untuk baris pertama
                        ]
                        final_cols = [c for c in display_cols if c in r_df.columns]
                        
                        # Nama sheet dibatasi 31 karakter
                        sheet_name = f"RUTE {r_id}"
                        r_df[final_cols].to_excel(writer, index=False, sheet_name=sheet_name)
                        
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white'})
                        for col_num, value in enumerate(final_cols):
                            worksheet.write(0, col_num, value, header_fmt)
                            worksheet.set_column(col_num, col_num, 20)

                    # 3. Sheet Overflow
                    if not overflow.empty:
                        overflow[['CUSTOMER NAME', 'ALAMAT', 'CUSTOMER CODE']].to_excel(writer, index=False, sheet_name='TIDAK TERCOVER')

                st.download_button(
                    label="📥 Download Excel Multi-Sheet",
                    data=buffer.getvalue(),
                    file_name=f"Jadwal_Audit_{datetime.date.today()}.xlsx",
                    mime="application/vnd.ms-excel",
                    type="primary"
                )