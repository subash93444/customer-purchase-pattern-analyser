import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from loader import load_data

# ── Data ──────────────────────────────────────────────────────────────────────
df = load_data()

# ── Theme ─────────────────────────────────────────────────────────────────────
PALETTE  = ['#FF9900','#00D4FF','#FF3366','#00FF88','#7B2FBE',
            '#FFD700','#F72585','#4CC9F0','#FFBE0B','#06D6A0']
DARK_BG  = '#070B14'
CARD_BG  = '#0D1526'
CARD2_BG = '#131F35'
AMAZON   = '#FF9900'
BLUE     = '#00D4FF'
TEXT     = '#E8EDF5'
MUTED    = '#5A6A85'
BORDER   = '#1A2840'

CL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Rajdhani, sans-serif', color=TEXT, size=12),
    margin=dict(l=10, r=10, t=38, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

def card(content, style=None):
    base = {'background': CARD_BG, 'border': f'1px solid {BORDER}',
            'borderRadius': '12px', 'padding': '14px'}
    if style: base.update(style)
    return html.Div(content, style=base)

def gchart(fig): return dcc.Graph(figure=fig, config={'displayModeBar': False})

def fmt(v):
    v = float(v)
    if v >= 1e7:  return f'₹{v/1e7:.1f}Cr'
    if v >= 1e5:  return f'₹{v/1e5:.1f}L'
    if v >= 1e3:  return f'₹{v/1e3:.1f}K'
    return f'₹{v:.0f}'

def kpi(icon, title, val, sub='', color=AMAZON):
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize':'24px'}),
            html.Div([
                html.P(title, style={'margin':0,'fontSize':'10px','color':MUTED,'letterSpacing':'1.5px','textTransform':'uppercase'}),
                html.H3(val,  style={'margin':'2px 0 0','color':color,'fontSize':'22px','fontWeight':'700','fontFamily':'Orbitron,sans-serif'}),
                html.P(sub,   style={'margin':0,'fontSize':'11px','color':MUTED}),
            ], style={'marginLeft':'12px'})
        ], style={'display':'flex','alignItems':'center'})
    ], style={
        'background': f'linear-gradient(135deg,{CARD_BG},{CARD2_BG})',
        'border': f'1px solid {BORDER}',
        'borderLeft': f'3px solid {color}',
        'borderRadius': '12px',
        'padding': '16px 20px',
        'flex': '1', 'minWidth': '165px'
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
NAV = [
    ('🏠','Overview',         '/'),
    ('📦','Product Analysis', '/products'),
    ('🏷️','Category Insights','/category'),
    ('💸','Discount & Price', '/discount'),
    ('⭐','Rating Analytics', '/ratings'),
    ('🔍','Deep Search',      '/search'),
    ('🤖','ML Clusters',      '/ml'),
    ('📊','Summary Table',    '/table'),
]

sidebar = html.Div([
    html.Div([
        html.Span('🛒', style={'fontSize':'28px'}),
        html.Div([
            html.H4('PurchaseIQ', style={'margin':0,'fontFamily':'Orbitron,sans-serif','fontSize':'15px','color':AMAZON}),
            html.P('Amazon Analytics', style={'margin':0,'fontSize':'9px','color':MUTED,'letterSpacing':'2px'}),
        ], style={'marginLeft':'10px'})
    ], style={'display':'flex','alignItems':'center','padding':'22px 18px 18px','borderBottom':f'1px solid {BORDER}'}),

    html.Div([
        dcc.Link(
            html.Div([
                html.Span(ic, style={'fontSize':'15px','width':'20px'}),
                html.Span(lb, style={'marginLeft':'10px','fontSize':'13px','fontWeight':'500'})
            ], style={'display':'flex','alignItems':'center','padding':'10px 18px',
                      'borderRadius':'8px','margin':'2px 8px','color':TEXT,'transition':'all .2s',
                      'cursor':'pointer'}),
            href=pg, style={'textDecoration':'none'}
        ) for ic,lb,pg in NAV
    ], style={'padding':'10px 0','flex':1}),

    html.Div([
        html.P(f'📁 {len(df):,} products loaded', style={'fontSize':'10px','color':MUTED,'textAlign':'center','letterSpacing':'1px'})
    ], style={'padding':'10px','borderTop':f'1px solid {BORDER}'}),
], style={'width':'215px','minHeight':'100vh','background':CARD_BG,
          'borderRight':f'1px solid {BORDER}','display':'flex','flexDirection':'column',
          'position':'fixed','top':0,'left':0,'zIndex':100})

# ── Filter bar ────────────────────────────────────────────────────────────────
def make_dd(id_, opts, ph, w='150px'):
    return dcc.Dropdown(id=id_, options=[{'label':o,'value':o} for o in opts],
        value=None, placeholder=ph, multi=True,
        style={'width':w,'fontSize':'12px','minWidth':w})

filter_bar = html.Div([
    html.Div([
        make_dd('f-cat',  sorted(df['main_category'].unique()), '🏷️ Category', '170px'),
        make_dd('f-subcat',sorted(df['sub_category'].unique()), '📂 Sub-Category','190px'),
        make_dd('f-tier', [str(t) for t in df['price_tier'].cat.categories], '💰 Price Tier','185px'),
        make_dd('f-dbucket',[str(b) for b in df['discount_bucket'].cat.categories],'💸 Discount','140px'),
        make_dd('f-rbucket',[str(r) for r in df['rating_bucket'].cat.categories],'⭐ Rating','140px'),
        html.Button('↺ Reset', id='btn-reset', n_clicks=0,
            style={'background':'transparent','border':f'1px solid {BORDER}','color':MUTED,
                   'borderRadius':'6px','padding':'4px 14px','cursor':'pointer','fontSize':'12px'}),
    ], style={'display':'flex','gap':'10px','alignItems':'center','flexWrap':'wrap'})
], style={'background':CARD_BG,'borderBottom':f'1px solid {BORDER}',
          'padding':'10px 22px','position':'sticky','top':0,'zIndex':50})

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        'https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='PurchaseIQ — Amazon')
server = app.server

app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    html.Div([
        filter_bar,
        html.Div(id='page-content', style={'padding':'22px'})
    ], style={'marginLeft':'215px','background':DARK_BG,'minHeight':'100vh'})
], style={'fontFamily':'Rajdhani,sans-serif','background':DARK_BG,'color':TEXT})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(d):
    total_prod   = len(d)
    avg_disc     = d['discount_percentage'].mean()
    avg_rating   = d['rating'].mean()
    avg_price    = d['discounted_price'].mean()
    total_rev    = d['revenue_proxy'].sum()
    top_cat      = d['main_category'].value_counts().idxmax()

    # Category donut
    cat_cnt = d['main_category'].value_counts().reset_index()
    cat_cnt.columns = ['Category','Count']
    donut = px.pie(cat_cnt, values='Count', names='Category', hole=0.62,
                   color_discrete_sequence=PALETTE)
    donut.update_traces(textinfo='percent+label', textfont_size=10,
                        marker=dict(line=dict(color=DARK_BG,width=2)))
    donut.update_layout(**CL, height=280, showlegend=False, title='Products by Category')

    # Avg discount per category
    d_cat = d.groupby('main_category')['discount_percentage'].mean().sort_values().reset_index()
    disc_bar = px.bar(d_cat, x='discount_percentage', y='main_category', orientation='h',
                      color='discount_percentage', color_continuous_scale=['#1A2840',AMAZON],
                      text=d_cat['discount_percentage'].round(1))
    disc_bar.update_traces(texttemplate='%{text}%', textposition='outside')
    disc_bar.update_layout(**CL, height=280, title='Avg Discount % by Category', coloraxis_showscale=False)

    # Rating distribution
    rat_hist = px.histogram(d, x='rating', nbins=20, color_discrete_sequence=[AMAZON])
    rat_hist.update_layout(**CL, height=240, title='Rating Distribution', bargap=0.05)
    rat_hist.update_traces(marker_line_color=DARK_BG, marker_line_width=1)

    # Price tier distribution
    tier_cnt = d['price_tier'].value_counts().sort_index().reset_index()
    tier_cnt.columns = ['Tier','Count']
    tier_bar = px.bar(tier_cnt, x='Tier', y='Count',
                      color='Count', color_continuous_scale=['#1A2840', BLUE])
    tier_bar.update_layout(**CL, height=240, title='Products by Price Tier', coloraxis_showscale=False)
    tier_bar.update_traces(marker_line_color=DARK_BG, marker_line_width=1)

    # Top 10 by popularity
    top10 = d.nlargest(10,'popularity_score')[['short_name','popularity_score','rating','rating_count']]
    pop_bar = px.bar(top10, x='popularity_score', y='short_name', orientation='h',
                     color='rating', color_continuous_scale=['#FF3366','#00FF88'],
                     hover_data={'rating':True,'rating_count':True,'popularity_score':':.1f'})
    pop_bar.update_layout(**CL, height=340, title='Top 10 Most Popular Products (log rating_count × rating)',
                          coloraxis_colorbar=dict(title='Rating'))

    # Sub-category count top 12
    sub_cnt = d['sub_category'].value_counts().head(12).reset_index()
    sub_cnt.columns = ['Sub','Count']
    sub_bar = px.bar(sub_cnt, x='Count', y='Sub', orientation='h',
                     color='Count', color_continuous_scale=['#1A2840','#7B2FBE'])
    sub_bar.update_layout(**CL, height=340, title='Top 12 Sub-Categories', coloraxis_showscale=False)

    return html.Div([
        # KPIs
        html.Div([
            kpi('📦','Total Products',   f'{total_prod:,}',    f'Across {d["main_category"].nunique()} categories', AMAZON),
            kpi('💸','Avg Discount',     f'{avg_disc:.1f}%',   'Off actual price',  '#FF3366'),
            kpi('⭐','Avg Rating',       f'{avg_rating:.2f}',  'Out of 5.0',        '#FFD700'),
            kpi('💰','Avg Disc. Price',  fmt(avg_price),       'After discount',    BLUE),
            kpi('🔥','Revenue Proxy',    fmt(total_rev),       'log(reviews)×price','#00FF88'),
            kpi('🏆','Top Category',     top_cat.replace('&',' & ')[:18], 'Most products', '#7B2FBE'),
        ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'20px'}),

        html.Div([card([gchart(donut)]), card([gchart(disc_bar)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        html.Div([card([gchart(rat_hist)]), card([gchart(tier_bar)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        html.Div([card([gchart(pop_bar)]), card([gchart(sub_bar)])],
                 style={'display':'flex','gap':'16px'}),
    ])


def page_products(d):
    # Top 15 by revenue proxy
    t15 = d.nlargest(15,'revenue_proxy')
    fig1 = px.bar(t15, x='revenue_proxy', y='short_name', orientation='h',
                  color='discounted_price', color_continuous_scale=['#7B2FBE',AMAZON],
                  hover_data={'product_name':True,'discounted_price':True,'rating':True})
    fig1.update_layout(**CL, height=380, title='Top 15 Products by Revenue Proxy', coloraxis_showscale=False)

    # Savings vs discounted price scatter
    fig2 = px.scatter(d, x='discounted_price', y='savings', color='main_category',
                      size='rating_count', size_max=25, opacity=0.75,
                      color_discrete_sequence=PALETTE,
                      hover_data={'product_name':True,'discount_percentage':True,'rating':True})
    fig2.update_layout(**CL, height=340, title='Savings vs Discounted Price (size = Rating Count)')

    # Price vs rating - only add trendline if enough data points
    if len(d) >= 30:
        fig3 = px.scatter(d, x='discounted_price', y='rating', color='main_category',
                          trendline='lowess', color_discrete_sequence=PALETTE, opacity=0.6,
                          hover_data={'product_name':True,'discount_percentage':True})
    else:
        fig3 = px.scatter(d, x='discounted_price', y='rating', color='main_category',
                          color_discrete_sequence=PALETTE, opacity=0.6,
                          hover_data={'product_name':True,'discount_percentage':True})
    fig3.update_layout(**CL, height=320, title='Price vs Rating (LOWESS trend per category)')

    # Top 10 highest savings
    top_sav = d.nlargest(10,'savings')
    fig4 = px.bar(top_sav, x='savings', y='short_name', orientation='h',
                  color='discount_percentage', color_continuous_scale=['#FFD700','#FF3366'],
                  text=top_sav['savings'].apply(fmt))
    fig4.update_traces(textposition='outside')
    fig4.update_layout(**CL, height=340, title='Top 10 Products by ₹ Savings', coloraxis_showscale=False)

    return html.Div([
        card([gchart(fig1)], {'marginBottom':'16px'}),
        html.Div([card([gchart(fig2)]), card([gchart(fig3)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig4)]),
    ])


def page_category(d):
    # Revenue proxy by main_category treemap
    cat_agg = d.groupby(['main_category','sub_category']).agg(
        revenue_proxy=('revenue_proxy','sum'),
        avg_rating=('rating','mean'),
        count=('product_id','count'),
        avg_discount=('discount_percentage','mean'),
    ).reset_index()

    fig1 = px.treemap(cat_agg, path=['main_category','sub_category'],
                      values='revenue_proxy', color='avg_rating',
                      color_continuous_scale='RdYlGn',
                      hover_data={'count':True,'avg_discount':':.1f','avg_rating':':.2f'})
    fig1.update_layout(**CL, height=400, title='Category → Sub-Category Revenue Treemap (color=Avg Rating)')

    # Category vs sub-category heatmap (avg discount)
    hm = d.groupby(['main_category','discount_bucket'])['product_id'].count().unstack(fill_value=0)
    fig2 = px.imshow(hm, color_continuous_scale='Cividis', aspect='auto', text_auto=True)
    fig2.update_layout(**CL, height=300, title='Product Count: Category × Discount Bucket')

    # Sub-cat avg rating top 15
    sub_rat = d.groupby('sub_category')['rating'].mean().nlargest(15).sort_values().reset_index()
    fig3 = px.bar(sub_rat, x='rating', y='sub_category', orientation='h',
                  color='rating', color_continuous_scale=['#FF3366','#00FF88'])
    fig3.update_layout(**CL, height=380, title='Top 15 Sub-Categories by Avg Rating', coloraxis_showscale=False)

    # Sunburst: main → price tier
    sb = d.groupby(['main_category','price_tier'])['product_id'].count().reset_index()
    sb.columns = ['main_category','price_tier','count']
    sb['price_tier'] = sb['price_tier'].astype(str)
    fig4 = px.sunburst(sb, path=['main_category','price_tier'], values='count',
                       color_discrete_sequence=PALETTE)
    fig4.update_layout(**CL, height=380, title='Category → Price Tier Sunburst')

    return html.Div([
        card([gchart(fig1)], {'marginBottom':'16px'}),
        html.Div([card([gchart(fig2)]), card([gchart(fig3)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig4)]),
    ])


def page_discount(d):
    # Discount bucket distribution
    db_cnt = d['discount_bucket'].value_counts().sort_index().reset_index()
    db_cnt.columns = ['Bucket','Count']
    db_cnt['Bucket'] = db_cnt['Bucket'].astype(str)
    fig1 = px.bar(db_cnt, x='Bucket', y='Count',
                  color='Count', color_continuous_scale=['#1A2840','#FF3366'],
                  text='Count')
    fig1.update_traces(textposition='outside')
    fig1.update_layout(**CL, height=260, title='Products by Discount Bucket', coloraxis_showscale=False)

    # Discount vs rating scatter - only add trendline if enough data points
    if len(d) >= 30:
        fig2 = px.scatter(d, x='discount_percentage', y='rating',
                          color='main_category', size='rating_count',
                          size_max=22, opacity=0.7, color_discrete_sequence=PALETTE,
                          trendline='ols', trendline_scope='overall',
                          hover_data={'product_name':True,'discounted_price':True})
    else:
        fig2 = px.scatter(d, x='discount_percentage', y='rating',
                          color='main_category', size='rating_count',
                          size_max=22, opacity=0.7, color_discrete_sequence=PALETTE,
                          hover_data={'product_name':True,'discounted_price':True})
    fig2.update_layout(**CL, height=340, title='Discount % vs Rating (OLS trend)')

    # Actual vs discounted price - handle small datasets
    sample_size = min(80, max(len(d), 10))
    sample = d.nlargest(sample_size, 'rating_count')
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Actual Price',     x=sample['short_name'], y=sample['actual_price'],    marker_color='#1A2840'))
    fig3.add_trace(go.Bar(name='Discounted Price', x=sample['short_name'], y=sample['discounted_price'],marker_color=AMAZON))
    fig3.update_layout(**CL, height=320, barmode='overlay', title=f'Top {sample_size} (by reviews): Actual vs Discounted Price',
                       xaxis_tickangle=-45, xaxis_showticklabels=False)

    # Avg discount per price tier
    pt_disc = d.groupby('price_tier', observed=True)['discount_percentage'].mean().reset_index()
    pt_disc.columns = ['Price Tier','Avg Discount %']
    pt_disc['Price Tier'] = pt_disc['Price Tier'].astype(str)
    fig4 = px.bar(pt_disc, x='Price Tier', y='Avg Discount %',
                  color='Avg Discount %', color_continuous_scale=['#00D4FF','#FF9900'],
                  text=pt_disc['Avg Discount %'].round(1))
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(**CL, height=280, title='Avg Discount % by Price Tier', coloraxis_showscale=False)

    # Box plot discount by category
    fig5 = px.box(d, x='main_category', y='discount_percentage',
                  color='main_category', color_discrete_sequence=PALETTE, points='outliers')
    fig5.update_layout(**CL, height=300, title='Discount Distribution by Category', showlegend=False)
    fig5.update_xaxes(tickangle=-30)

    return html.Div([
        html.Div([card([gchart(fig1)]), card([gchart(fig4)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig2)], {'marginBottom':'16px'}),
        card([gchart(fig3)], {'marginBottom':'16px'}),
        card([gchart(fig5)]),
    ])


def page_ratings(d):
    # Rating bucket donut
    rb = d['rating_bucket'].value_counts().sort_index().reset_index()
    rb.columns = ['Bucket','Count']; rb['Bucket'] = rb['Bucket'].astype(str)
    fig1 = px.pie(rb, values='Count', names='Bucket', hole=0.55,
                  color_discrete_sequence=['#FF3366','#FF9900','#FFD700','#00FF88','#00D4FF'])
    fig1.update_layout(**CL, height=280, title='Rating Bucket Distribution')

    # Rating count vs rating
    fig2 = px.scatter(d, x='rating_count', y='rating', color='main_category',
                      log_x=True, size='discount_percentage', size_max=20, opacity=0.7,
                      color_discrete_sequence=PALETTE,
                      hover_data={'product_name':True,'discounted_price':True})
    fig2.update_layout(**CL, height=340, title='Rating Count (log) vs Rating (size=Discount %)')

    # Category avg rating bar
    cat_rat = d.groupby('main_category')['rating'].mean().sort_values().reset_index()
    fig3 = px.bar(cat_rat, x='rating', y='main_category', orientation='h',
                  color='rating', color_continuous_scale=['#FF3366','#00FF88'],
                  text=cat_rat['rating'].round(2))
    fig3.update_traces(textposition='outside')
    fig3.update_layout(**CL, height=280, title='Avg Rating by Category', coloraxis_showscale=False)

    # Value score (rating × discount factor) vs price
    fig4 = px.scatter(d, x='discounted_price', y='value_score',
                      color='main_category', size='rating_count',
                      size_max=20, opacity=0.75, color_discrete_sequence=PALETTE,
                      hover_data={'product_name':True,'rating':True,'discount_percentage':True})
    fig4.update_layout(**CL, height=320, title='Value Score vs Price (score = rating × discount_bonus)')

    # Top 10 highest rated (min 1000 reviews)
    top_rated = d[d['rating_count'] >= 1000].nlargest(10,'rating')
    fig5 = px.bar(top_rated, x='rating', y='short_name', orientation='h',
                  color='rating_count', color_continuous_scale=['#1A2840',AMAZON],
                  text='rating')
    fig5.update_traces(texttemplate='%{text:.1f}★', textposition='outside')
    fig5.update_layout(**CL, height=320, title='Top 10 Highest Rated (≥1,000 reviews)', coloraxis_showscale=False)

    # Sub-cat rating heatmap
    sr_hm = d.groupby(['main_category','rating_bucket'], observed=True)['product_id'].count().unstack(fill_value=0)
    sr_hm.columns = [str(c) for c in sr_hm.columns]
    fig6 = px.imshow(sr_hm, color_continuous_scale='Blues', aspect='auto', text_auto=True)
    fig6.update_layout(**CL, height=280, title='Category × Rating Bucket Heatmap')

    return html.Div([
        html.Div([card([gchart(fig1)]), card([gchart(fig3)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig2)], {'marginBottom':'16px'}),
        html.Div([card([gchart(fig4)]), card([gchart(fig5)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig6)]),
    ])


def page_search(d):
    return html.Div([
        card([
            html.P('🔍 PRODUCT DEEP SEARCH', style={'margin':'0 0 10px','color':MUTED,'fontSize':'10px','letterSpacing':'2px'}),
            dcc.Input(id='search-input', placeholder='Search product name or category…',
                      debounce=True, style={
                          'width':'100%','background':CARD2_BG,'border':f'1px solid {BORDER}',
                          'color':TEXT,'borderRadius':'8px','padding':'10px 14px','fontSize':'14px',
                          'outline':'none','boxSizing':'border-box'
                      }),
            html.Div(id='search-results', style={'marginTop':'16px'})
        ])
    ])


def page_ml(d):
    feats = d[['discounted_price','actual_price','discount_percentage','rating','rating_count','savings','value_score']].dropna()
    idx   = feats.index
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    ml_df = d.loc[idx].copy()
    ml_df['Cluster']  = labels.astype(str)
    ml_df['PC1'], ml_df['PC2'] = coords[:,0], coords[:,1]

    cluster_names = {
        '0':'Budget Basics','1':'Mid-Range Value',
        '2':'Premium Picks','3':'Deep Discount','4':'High Demand'
    }
    ml_df['Cluster_Label'] = ml_df['Cluster'].map(cluster_names)

    fig1 = px.scatter(ml_df, x='PC1', y='PC2', color='Cluster_Label',
                      color_discrete_sequence=PALETTE, opacity=0.75,
                      hover_data={'product_name':True,'discounted_price':True,
                                  'discount_percentage':True,'rating':True,'Cluster':False})
    fig1.update_layout(**CL, height=400, title='K-Means Customer/Product Clusters (PCA 2D)')

    # Cluster profiling
    prof = ml_df.groupby('Cluster_Label').agg(
        Count=('product_id','count'),
        Avg_Price=('discounted_price','mean'),
        Avg_Discount=('discount_percentage','mean'),
        Avg_Rating=('rating','mean'),
        Avg_Reviews=('rating_count','mean'),
    ).reset_index()

    fig2 = px.bar(prof, x='Cluster_Label', y='Avg_Price',
                  color='Cluster_Label', color_discrete_sequence=PALETTE, text='Count')
    fig2.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig2.update_layout(**CL, height=300, title='Avg Discounted Price by Cluster', showlegend=False)

    fig3 = px.scatter(ml_df, x='discounted_price', y='rating',
                      color='Cluster_Label', color_discrete_sequence=PALETTE, opacity=0.65,
                      size='rating_count', size_max=20,
                      hover_data={'product_name':True,'discount_percentage':True})
    fig3.update_layout(**CL, height=340, title='Cluster: Price vs Rating (size=Reviews)')

    # Radar chart cluster profiles
    cats_radar = ['Avg_Price','Avg_Discount','Avg_Rating','Avg_Reviews']
    prof_norm  = prof.copy()
    for c in cats_radar:
        mn, mx = prof[c].min(), prof[c].max()
        prof_norm[c] = (prof[c] - mn) / (mx - mn + 1e-9)

    fig4 = go.Figure()
    for i, row in prof_norm.iterrows():
        fig4.add_trace(go.Scatterpolar(
            r=[row[c] for c in cats_radar] + [row[cats_radar[0]]],
            theta=cats_radar + [cats_radar[0]],
            name=row['Cluster_Label'],
            line_color=PALETTE[i], fill='toself', opacity=0.4
        ))
    fig4.update_layout(**CL, height=360, title='Cluster Radar Profile (normalised)',
                       polar=dict(bgcolor='rgba(0,0,0,0)',
                                  radialaxis=dict(gridcolor=BORDER,color=MUTED),
                                  angularaxis=dict(gridcolor=BORDER,color=TEXT)))

    return html.Div([
        card([gchart(fig1)], {'marginBottom':'16px'}),
        html.Div([card([gchart(fig2)]), card([gchart(fig3)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([gchart(fig4)]),
    ])


def page_table(d):
    cols = ['product_name','main_category','sub_category',
            'discounted_price','actual_price','discount_percentage',
            'rating','rating_count','price_tier','discount_bucket']
    show = d[cols].copy()
    show['discount_percentage'] = show['discount_percentage'].apply(lambda x: f'{x:.0f}%')
    show['discounted_price']    = show['discounted_price'].apply(lambda x: f'₹{x:,.0f}')
    show['actual_price']        = show['actual_price'].apply(lambda x: f'₹{x:,.0f}')
    show['price_tier']          = show['price_tier'].astype(str)
    show['discount_bucket']     = show['discount_bucket'].astype(str)
    show = show.rename(columns={
        'product_name':'Product','main_category':'Category','sub_category':'Sub-Cat',
        'discounted_price':'Disc. Price','actual_price':'Actual Price',
        'discount_percentage':'Discount','rating':'Rating','rating_count':'Reviews',
        'price_tier':'Price Tier','discount_bucket':'Disc. Bucket'
    })
    tbl = dash_table.DataTable(
        data=show.to_dict('records'),
        columns=[{'name':c,'id':c} for c in show.columns],
        page_size=20, filter_action='native', sort_action='native',
        style_table={'overflowX':'auto'},
        style_header={'backgroundColor':CARD2_BG,'color':AMAZON,'fontWeight':'700',
                      'fontFamily':'Rajdhani','fontSize':'12px','border':f'1px solid {BORDER}'},
        style_cell={'backgroundColor':CARD_BG,'color':TEXT,'fontFamily':'Rajdhani',
                    'fontSize':'12px','border':f'1px solid {BORDER}','padding':'7px 10px',
                    'maxWidth':'220px','overflow':'hidden','textOverflow':'ellipsis'},
        style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':CARD2_BG}],
        tooltip_data=[{c:{'value':str(row[c]),'type':'markdown'} for c in show.columns} for row in show.to_dict('records')],
        tooltip_duration=None,
    )
    return html.Div([
        card([
            html.P('📊 FULL PRODUCT TABLE — filterable & sortable',
                   style={'color':MUTED,'fontSize':'10px','letterSpacing':'2px','margin':'0 0 12px'}),
            tbl
        ])
    ])


# ── Page titles ───────────────────────────────────────────────────────────────
TITLES = {
    '/':          '🏠 Dashboard Overview',
    '/products':  '📦 Product Analysis',
    '/category':  '🏷️ Category Insights',
    '/discount':  '💸 Discount & Price Intelligence',
    '/ratings':   '⭐ Rating Analytics',
    '/search':    '🔍 Deep Product Search',
    '/ml':        '🤖 ML Cluster Analysis',
    '/table':     '📊 Full Product Table',
}

PAGE_FN = {
    '/':         page_overview,
    '/products': page_products,
    '/category': page_category,
    '/discount': page_discount,
    '/ratings':  page_ratings,
    '/search':   page_search,
    '/ml':       page_ml,
    '/table':    page_table,
}

# ── Callbacks ─────────────────────────────────────────────────────────────────
def apply_filters(years, cats, subcat, tier, dbucket, rbucket):
    d = df.copy()
    if cats:    d = d[d['main_category'].isin(cats)]
    if subcat:  d = d[d['sub_category'].isin(subcat)]
    if tier:    d = d[d['price_tier'].astype(str).isin(tier)]
    if dbucket: d = d[d['discount_bucket'].astype(str).isin(dbucket)]
    if rbucket: d = d[d['rating_bucket'].astype(str).isin(rbucket)]
    return d

@app.callback(
    Output('page-content','children'),
    Input('url','pathname'),
    Input('f-cat','value'),
    Input('f-subcat','value'),
    Input('f-tier','value'),
    Input('f-dbucket','value'),
    Input('f-rbucket','value'),
)
def render(pathname, cats, subcat, tier, dbucket, rbucket):
    d = apply_filters(None, cats, subcat, tier, dbucket, rbucket)
    pathname = pathname or '/'
    title = TITLES.get(pathname,'Dashboard')
    header = html.Div([
        html.H2(title, style={'margin':0,'fontFamily':'Orbitron,sans-serif','fontSize':'17px','color':TEXT}),
        html.P(f'{len(d):,} products  •  {d["main_category"].nunique()} categories  •  Avg rating {d["rating"].mean():.2f}★',
               style={'margin':'4px 0 0','fontSize':'11px','color':MUTED})
    ], style={'marginBottom':'18px','paddingBottom':'14px','borderBottom':f'1px solid {BORDER}'})
    if len(d) == 0:
        return html.Div([header, html.P('No products match the selected filters.', style={'color':MUTED})])
    fn = PAGE_FN.get(pathname, page_overview)
    return html.Div([header, fn(d)])


@app.callback(
    Output('f-cat','value'), Output('f-subcat','value'),
    Output('f-tier','value'), Output('f-dbucket','value'), Output('f-rbucket','value'),
    Input('btn-reset','n_clicks')
)
def reset(_): return None, None, None, None, None


@app.callback(
    Output('search-results','children'),
    Input('search-input','value'),
)
def search_products(query):
    if not query or len(query.strip()) < 2:
        return html.P('Type at least 2 characters to search…', style={'color':MUTED,'fontSize':'12px'})
    q = query.strip().lower()
    mask = (df['product_name'].str.lower().str.contains(q, na=False) |
            df['main_category'].str.lower().str.contains(q, na=False) |
            df['sub_category'].str.lower().str.contains(q, na=False))
    res = df[mask].head(30)
    if len(res) == 0:
        return html.P('No products found.', style={'color':MUTED})

    # Charts for search results
    fig1 = px.bar(res.nlargest(10,'rating_count'), x='rating_count', y='short_name', orientation='h',
                  color='rating', color_continuous_scale=['#FF3366','#00FF88'],
                  hover_data={'product_name':True,'discounted_price':True})
    fig1.update_layout(**CL, height=300, title=f'Top 10 results by reviews — "{query}"', coloraxis_showscale=False)

    fig2 = px.scatter(res, x='discounted_price', y='rating',
                      size='rating_count', color='main_category',
                      color_discrete_sequence=PALETTE, size_max=25, opacity=0.8,
                      hover_data={'product_name':True,'discount_percentage':True})
    fig2.update_layout(**CL, height=300, title='Search Results — Price vs Rating')

    rows = []
    for _, r in res.iterrows():
        rows.append(html.Div([
            html.Div([
                html.P(r['product_name'][:80]+'…' if len(r['product_name'])>80 else r['product_name'],
                       style={'margin':0,'fontSize':'13px','fontWeight':'600','color':TEXT}),
                html.P(f"{r['main_category']} › {r['sub_category']}",
                       style={'margin':'2px 0 0','fontSize':'11px','color':MUTED}),
            ], style={'flex':3}),
            html.Div([
                html.Span(f"₹{r['discounted_price']:,.0f}", style={'color':AMAZON,'fontWeight':'700','fontSize':'14px'}),
                html.Span(f" (was ₹{r['actual_price']:,.0f})", style={'color':MUTED,'fontSize':'11px','textDecoration':'line-through'}),
                html.Span(f" -{r['discount_percentage']:.0f}%", style={'color':'#FF3366','fontSize':'11px','marginLeft':'6px'}),
            ], style={'flex':2,'textAlign':'right'}),
            html.Div([
                html.Span(f"{'★'*round(r['rating'])} {r['rating']:.1f}",
                          style={'color':'#FFD700','fontSize':'13px'}),
                html.P(f"{int(r['rating_count']):,} reviews",
                       style={'margin':'2px 0 0','fontSize':'10px','color':MUTED}),
            ], style={'flex':1,'textAlign':'right'}),
        ], style={'display':'flex','alignItems':'center','gap':'12px','padding':'10px 0',
                  'borderBottom':f'1px solid {BORDER}'}))

    return html.Div([
        html.P(f'Found {len(res)} products matching "{query}"',
               style={'color':MUTED,'fontSize':'11px','marginBottom':'12px'}),
        html.Div([card([gchart(fig1)]), card([gchart(fig2)])],
                 style={'display':'flex','gap':'16px','marginBottom':'16px'}),
        card([html.Div(rows)])
    ])


if __name__ == '__main__':
    print('\n' + '═'*58)
    print('  🛒  Amazon PurchaseIQ  →  http://127.0.0.1:8050')
    print('═'*58 + '\n')
    app.run(debug=True, port=8050)
