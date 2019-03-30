# -*- coding: utf-8 -*-

import os
import pymysql as sql
import numpy as np
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
#import dash_bootstrap_components as dbc
import dash_table
import dateutil
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime as dt
from datetime import date
from datetime import timedelta
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
#import dotenv
import os
import flask 
import urllib.parse as urlparse

DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
USERNAME = os.getenv("USERNAME")
DASHBOARDPASS = os.getenv("DASHBOARDPASS")
# Keep this out of source code repository - save in a file or a database
# VALID_USERNAME_PASSWORD_PAIRS = [
#     ['USERNAME', 'DASHBOARDPASS']
# ]

#Login and pass for dashboard
VALID_USERNAME_PASSWORD_PAIRS = [
    ['fla', 'test']
]
#print(DATABASE_USER)

app = dash.Dash(__name__)#, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.title = 'Dashboard - Fastlane Automotive'

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


##############################################################
#                                                            #
#             D  A  T  A     L  O  A  D  I  N  G             #
#                                                            #
##############################################################

wd = os.getcwd()
df = pd.read_csv('./finance.csv', sep = ';', parse_dates=True)
#df = pd.read_csv('/home/shiny/dashboards/dashboard_data_supply/data_extracts/finance.csv', sep = ';', parse_dates=True)

dfcol = ['Kanal', 'Einkaufspreis', 'Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag', 
              'Anzahl Bestellungen', 'Wareneinsatz', 'Anzahl Artikelstück', 'Anzahl Bundle', 
              'Nettoumsatz', 'RE1', 'RE2']

df = df[dfcol]

df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

df.columns = ['Kanal', 'Einkaufspreis', 'Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag', 
              'Anzahl Bestellungen', 'Wareneinsatz', 'Anzahl Artikelstück', 'Anzahl Bundle', 
              'Nettoumsatz', 'RE1', 'RE2']
        
intcol = ['Anzahl Bestellungen', 'Anzahl Artikelstück', 'Anzahl Bundle']
df[intcol] = df[intcol].round(0)

# Aggegrierte Daten 
agg_dict = {'Anzahl Bestellungen':'sum', 'Wareneinsatz':'sum','Anzahl Artikelstück':'sum', 
            'Anzahl Bundle':'sum', 'Nettoumsatz':'sum', 'RE1':'sum', 'RE2':'sum'}

# Define Nettowarenkorb und Marge

df = df.assign(Nettowarenkorb = round(df['Nettoumsatz'] / df['Anzahl Bestellungen'],2)) 
df = df.assign(Marge = (100*(df['Nettoumsatz'] - df['Wareneinsatz'])  / df['Nettoumsatz']).round(2))

def cal_marge(dataframe):
    dataframe = dataframe.assign(Nettowarenkorb = round(dataframe['Nettoumsatz'] / dataframe['Anzahl Bestellungen'],2)) 
    dataframe = dataframe.assign(Marge = (100*(dataframe['Nettoumsatz'] - dataframe['Wareneinsatz'])  / dataframe['Nettoumsatz']).round(2))
    return dataframe

def group_unternehmen(dataframe):
    dataframe = dataframe.drop(['Kanal'], axis=1).groupby(['Einkaufspreis', 'Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag']).agg(agg_dict).reset_index()
    return dataframe

fladf = df.drop(['Kanal'], axis=1).groupby(['Einkaufspreis', 'Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag']).agg(agg_dict).reset_index()
fladf = cal_marge(fladf)
fladf.insert(0, 'Kanal', 'FLA')

teufeldf = group_unternehmen(df[df.Kanal.isin(['teufel-shop', 'teufel-ebay', 'teufel-tyre24', 'teufel-daparto'])])
teufeldf = cal_marge(teufeldf)
teufeldf.insert(0, 'Kanal', 'TTA')

prdf = group_unternehmen(df[df.Kanal.isin(['partsrunner-ebay', 'partsrunner-tyre24', 'partsrunner-daparto'])])
prdf = cal_marge(prdf)
prdf.insert(0, 'Kanal', 'Partsrunner')

ebaydf = group_unternehmen(df[df.Kanal.isin(['teufel-ebay', 'partsrunner-ebay', 'carmaster-ebay'])])
ebaydf = cal_marge(ebaydf)
ebaydf.insert(0, 'Kanal', 'Ebay')

dapartodf = group_unternehmen(df[df.Kanal.isin(['teufel-daparto', 'partsrunner-daparto'])])
dapartodf = cal_marge(dapartodf)
dapartodf.insert(0, 'Kanal', 'Daparto')

tyredf = group_unternehmen(df[df.Kanal.isin(['teufel-tyre24', 'partsrunner-tyre24'])])
tyredf = cal_marge(tyredf)
tyredf.insert(0, 'Kanal', 'Tyre24')

marktplatzdf = group_unternehmen(df[df.Kanal.isin(['teufel-daparto', 'partsrunner-daparto', 'teufel-tyre24', 'partsrunner-tyre24', 'teufel-ebay', 'partsrunner-ebay', 'carmaster-ebay'])])
marktplatzdf = cal_marge(marktplatzdf)
marktplatzdf.insert(0, 'Kanal', 'Marktplätze')

unternehmendf = fladf.append([teufeldf, prdf, ebaydf, dapartodf, tyredf, marktplatzdf], ignore_index=True) 

rohdaten = df.append(unternehmendf, ignore_index=True) 

rohdaten.iloc[:,7:] = rohdaten.iloc[:,7:].apply(lambda x: round(x, 2))

#Konstanten
METRICS = rohdaten.columns[7:]
CHANNELS = rohdaten.Kanal.unique()
EKPREIS = rohdaten.Einkaufspreis.unique()
COLORS = ['#7DFB6D', '#C7B815', '#D4752E', '#C7583F']


default_shop = ['FLA']
default_metric = 'Nettoumsatz'
default_ek = EKPREIS

exportfilename = './downloads/export-data.csv'
app.config['suppress_callback_exceptions']=True

##############################################################
#                                                            #
#                   L  A  Y  O  U  T                         #
#                                                            #
##############################################################

navbar = html.Div([
        html.Span(
            children=['Fastlane KPI Dashboard'], className = 'title'
        ),
        html.Div([
            html.A([
                html.Img(src='https://www.fastlane-automotive.de/wp-content/themes/fastlane/dev-Images/fastlane-logo.svg')
            ], target="_blank", rel="noopener noreferrer", href = 'https://www.fastlane-automotive.de/')
        ], id = "imgdiv")
    ], id = "titlediv"
)

datadiv = html.Div([
        # Links
        html.Div([
            ### Kanal 1
            dcc.Dropdown(
                id='channel1',
                options=[{'label': i, 'value': i} for i in sorted(CHANNELS)],
                multi=True,
                value = default_shop,
                placeholder = "Kanal auswählen",
                clearable = True,
                className = 'dropdown'
            ),
            ### KPI 1
            dcc.Dropdown(
                id='metric1',
                options=[{'label': i, 'value': i} for i in sorted(METRICS)],
                multi=False,
                value = default_metric,
                placeholder = "KPI 1 auswählen (Default: Nettoumsatz)",
                className = 'dropdown',
                clearable = True,
            ),
            ### Einkaufspreis
            dcc.Dropdown(
                id='ekpreis',
                options=[{'label': i, 'value': i} for i in sorted(EKPREIS)],
                multi=True,
                value = 'Einkaufspreiskategorie auswählen',
                placeholder = "Einkaufspreiskategorie auswählen",
                className = 'dropdown',
                clearable = True,
            )], id = "subdiv1", className = "subdiv"
        ),
        # Mitte
        html.Div([
            # Zeitraum auswählen
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=dt(2018, 1, 1),
                max_date_allowed=max(rohdaten.Datum).date(),
                initial_visible_month=dt.today().date().replace(day=1),
                display_format='DD/MM/YYYY',
                number_of_months_shown = 3,
                clearable = False,
                start_date=dt.today().date().replace(day=1),
                end_date=max(rohdaten.Datum).date(),
                first_day_of_week = 1
                #start_date=dt(2019, 2, 11).date(),
                #end_date=dt(2019, 2, 13).date(),
            ),
            # Group by
            html.Div([
                html.Div([dcc.Slider(
                    id = "groupslider",
                    min=1,
                    max=4,
                    marks={1: 'Tag',2: 'Monat',3: 'Quartal', 4: 'Jahr'},
                    value=1
                )], id = 'sliderdiv'),
                html.Div([dcc.Dropdown(
                    id = "vergleichszeitraum",
                    options=[{'label': i, 'value': i} for i in ['Vorjahr', 'Vormonat', 'Vorwoche']],
                    multi=False,
                    className = 'dropdown',
                    placeholder = "Vergleichszeitraum"
                )], id='vergleichszeitraumdiv'),
                html.Div([dcc.Checklist(
                    id = 'vergleichcheckbox',
                    options=[
                        {'label': 'Absolute Differenz', 'value': 'abs'},
                        {'label': 'Prozentuale Differenz', 'value': 'pro'},
                    ],
                    values=[]
                )], id = 'vergleichcheckboxdiv')
            ], id = "timediv"),
            # 
        ], id = "subdiv2", className = 'subdiv'  
        ),
        # Rechts
        html.Div([
            ### KPI 2
            dcc.Dropdown(
                id='metric2',
                options=[{'label': i, 'value': i} for i in METRICS],
                multi=False,
                clearable = True,
                placeholder = "KPI 2 auswählen",
                className = 'dropdown'
            ),
            ### Kanal 2
            dcc.Dropdown(
                id='channel2',
                options=[{'label': i, 'value': i} for i in CHANNELS],
                multi=True,
                clearable = False,
                placeholder = "Kanal 2 auswählen (noch nicht implementiert)",
                className = 'dropdown'
            )
            ], id = "subdiv3", className = "subdiv"
        )], 
    id = "datadiv"
)

graphdiv = html.Div([
            dcc.Graph(
            id='kpigraph',
            style={'marginTop': '0px','marginBottom' : '20px','fontFamily': 'sans-serif','fontSize' : '14px','height' : '620px', 'width':'100%'},),
            #html.Div([
            html.A(id='download-link', children = [html.Button('Daten als CSV herunterladen', id='export-button', n_clicks = 0)], 
                    download = True, href = "./downloads/export-data.csv"),
        ], 
    id = "graphdiv"
)

tablediv = html.Div([dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in rohdaten[['Kanal', 'Datum', default_metric]].columns],
            #data=rohdaten.to_dict("rows"),
            editable=True,
            
            style_header={'backgroundColor': '#2a3f5f', 'fontSize': '1.2em', 'fontFamily': 'sans-serif', 'fontWeight': 'bold', 'color': 'white',},

            style_cell={'textAlign': 'center','fontSize': '1em','fontFamily': 'sans-serif','width': '20px', 'whiteSpace': 'no-wrap', 'paddingLeft' : '10px',},

            style_cell_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'
            }],
        )], id = "tablediv")

tab1 = dcc.Tab(label = "Fastlane KPI Dashboard", children = [
            html.Div(children=[
                #navbar,
                html.Div([
                    datadiv, graphdiv, tablediv
                ], id = 'mittediv')
            ], className = "bigdiv")
        ], className = "tab", selected_style = {"color" : "white", "background-color": "#2a3f5f", "font-weight": "bold"}, 
                              disabled_style = {"color" : "#2a3f5f", "background-color": "white"})

beiratdaten = dcc.DatePickerRange(
                id='date-picker-beirat',
                min_date_allowed=dt(2018, 1, 1),
                max_date_allowed=max(rohdaten.Datum).date(),
                initial_visible_month=dt.today().date().replace(day=1).replace(month=1),
                display_format='DD/MM/YYYY',
                number_of_months_shown = 3,
                clearable = False,
                start_date=dt.today().date().replace(day=1).replace(month=1),
                end_date=max(rohdaten.Datum).date(),
                first_day_of_week = 1
            )

beirat1 = html.Div([
            # html.Div([dash_table.DataTable(
            #         id='beirat-tabelle-1',
            #         columns=[{"name": i, "id": i} for i in rohdaten[['Kanal', 'Datum', default_metric]].columns],
            #         #data=rohdaten.to_dict("rows"),
            #         editable=True,
            #         style_header={'backgroundColor': '#2a3f5f', 'fontSize': '1.2em', 'fontFamily': 'sans-serif', 'fontWeight': 'bold', 'color': 'white',},
            #         style_cell={'textAlign': 'center','fontSize': '1em','fontFamily': 'sans-serif', 'whiteSpace': 'no-wrap',},
            #         style_cell_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'
            #         }],
            #     )], id = "beirat-tabelle-div"),
            html.Div([
                    html.Div([dcc.Graph(id='graph1',
                                        style={'marginTop': '0px','marginBottom' : '20px','fontFamily': 'sans-serif','fontSize' : '14px','height' : '300px'})], 
                            id = "graph1div", className = "beirat-graph"),
                ], id = "beirat-graphen-div")
            ], id = "beirat-graph-1", className = "beirat")

tab2 = dcc.Tab(label = "Beiratssitzung KPI", children = [
            html.Div(children=[
                html.Div([
                   beiratdaten,
                ], id = 'date-picker-beirat-div'),
                html.Div([
                    html.Button('Beiratsdaten generieren', id='beirat-create', n_clicks = 0)
                ], id = 'beirat-button-div')
            ], className = "bigdiv"),
            html.Div(children = [
                    beirat1
                    ]
                )
        ], className = "tab", selected_style = {"color" : "white", "background-color": "#2a3f5f", "font-weight": "bold"}, 
                              disabled_style = {"color" : "#2a3f5f", "background-color": "white"})

app.layout = html.Div([
    dcc.Tabs(id = "tabs", children = [
        tab1,
        tab2
    ], style = {"font-size": "2em", "height" : "70px"}, colors = {"border": "white", "primary": "gold"})
], id = "tabsdiv")




##############################################################
#                                                            #
#           Selbst definierte Funktionen                     #
#                                                            #
##############################################################

def scale_axes_values(data, metric1, metric2, l, vergleichszeitraum, vergleichcheckbox):
    gr1 = ['Nettoumsatz', 'RE1', 'RE2', 'Wareneinsatz']
    gr2 = ['Anzahl Artikelstück', 'Anzahl Bestellungen', 'Anzahl Bundle']

    if (metric1 in gr1 and metric2 in gr1) or (metric1 in gr2 and metric2 in gr2):

        if min(data) < 0:
            y_range = [min(data)*1.1, max(data)*1.1] 

        else:
            y_range = [min(data) * 0.6, max(data)*1.1] 

        l['yaxis']['range'] = y_range 
        l['yaxis2']['range'] = y_range 

    if vergleichcheckbox == ['pro'] and not (vergleichszeitraum is None or vergleichszeitraum == []):
        if metric2 is None or metric2 == [] or metric2 == metric1:
            l['yaxis']['ticksuffix'] = "%"
        else:
            l['yaxis']['ticksuffix'] = "%"
            l['yaxis2']['ticksuffix'] = "%"


def vergleichsvariable(vergleichszeitraum, start_date, end_date):
    if vergleichszeitraum == "Vorjahr":
        last_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    elif vergleichszeitraum == "Vormonat":
        last_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    elif vergleichszeitraum == "Vorwoche":
        last_start_date = pd.to_datetime(start_date) - pd.DateOffset(weeks=1)

    date_range = abs(dt.strptime(start_date, '%Y-%m-%d') - dt.strptime(end_date, '%Y-%m-%d')).days
    last_end_date = last_start_date + timedelta(date_range)
    last_start_date = last_start_date.strftime('%Y-%m-%d')
    last_end_date = last_end_date.strftime('%Y-%m-%d')

    return last_start_date, last_end_date


def filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum):

    # Für jeden Kanal, für jeden EK: 
    #   update table data
    #   update plot data
    #   plot layout

    ts = ''
    ts2 = ''
    traces = []
    lout = []

    currency_cols = ['Nettoumsatz', 'RE1', 'RE2', 'Nettowarenkorb', 'Wareneinsatz']
    
    # Layout format
    if metric1 in currency_cols:
        ts = "€"
    elif metric1 in ['Marge']:
        ts = "%" 
    if metric2 in currency_cols:
        ts2 = "€"
    elif metric2 in ['Marge']:
        ts2 = "%" 

    f = dict(family='sans-serif', size=14)

    y=dict(title = '<b>'+metric1+'</b>', titlefont = dict(size = 20), ticksuffix = ts, tickformat = " .2", zeroline=False, automargin = True, hoverformat = ' .2f')

    if not (metric2 is None or metric2 == []):
        y2 = dict(showgrid = False, title = '<b>'+metric2+'</b>', overlaying = 'y', side = 'right', titlefont = dict(size = 20), ticksuffix = ts2, tickformat = " .2", zeroline=False
        , hoverformat = ' .2f')

    # Filtern data nach EK
    if ek == "Einkaufspreiskategorie auswählen" or ek is None or ek == []: # Falls EK leer ist

        data_alltime = rohdaten[(rohdaten.Kanal == c)]
        data_alltime = data_alltime.drop(['Einkaufspreis'], axis = 1)

        data_aktuell = data_alltime[(data_alltime.Datum >= start_date) & (data_alltime.Datum <= end_date)]

        ekcol = []

        ek = ''

    else: # Falls EK ausgewählt wird

        data_alltime = rohdaten[(rohdaten.Kanal == c) & (rohdaten.Einkaufspreis == ek)]

        data_aktuell = data_alltime[(data_alltime.Datum >= start_date) & (data_alltime.Datum <= end_date)]

        ekcol = ['Einkaufspreis']


    # Aktuelle Daten: immer da, unabhängig von Vergleichszeitraum

    zeitlist = ['Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag']

    def groupslider_filter_data(data, zeit):
        zeitlist.remove(zeit)
        newdata = data.drop(zeitlist, axis = 1).groupby(['Kanal'] + ekcol + [zeit]).agg(agg_dict).reset_index()
        newdata = cal_marge(newdata)
        return newdata

    if groupslider == 1:
        zeit = 'Datum'
        data_aktuell = groupslider_filter_data(data_aktuell, zeit)
        format_time_data(data_aktuell, groupslider)

        x=dict(title = '<b>Datum</b>', showgrid = False, titlefont = dict(size = 20), type = 'category')
        x2=dict(showgrid = False, type = 'category', overlaying = 'x', side = 'top', titlefont = dict(size = 20))

    elif groupslider == 2:
        zeit = 'Monat'
        data_aktuell = groupslider_filter_data(data_aktuell, zeit)
        format_time_data(data_aktuell, groupslider)
        x=dict(title = '<b>Monat</b>', showgrid = False, titlefont = dict(size = 20), type = 'category')
        x2=dict(showgrid = False, type = 'category', overlaying = 'x', side = 'top', titlefont = dict(size = 20))

    elif groupslider == 3:
        zeit = 'Quartal'
        data_aktuell = groupslider_filter_data(data_aktuell, zeit)
        format_time_data(data_aktuell, groupslider)
        x=dict(title = '<b>Quartal</b>', showgrid = False, titlefont = dict(size = 20), type = 'category')
        x2=dict(showgrid = False, type = 'category', overlaying = 'x', side = 'top', titlefont = dict(size = 20))

    elif groupslider == 4:
        zeit = 'Jahr'
        data_aktuell = groupslider_filter_data(data_aktuell, zeit)
        format_time_data(data_aktuell, groupslider)

        x=dict(title = '<b>Jahr</b>', showgrid = False, titlefont = dict(size = 20), type = 'category')
        x2=dict(showgrid = False, overlaying = 'x', side = 'top', titlefont = dict(size = 20), type = 'category')

    data_aktuell.iloc[:,7:] = data_aktuell.iloc[:,7:].apply(lambda x: round(x, 2))

    cols = ['Kanal'] + ekcol + [zeit]

    if data_aktuell.shape[0] <= 22:
        m = {'l': 90, 'b': 10, 't': 20, 'r': 30}
        l = dict(orientation="h", x=0, y=-0.2)
    else:
        m = {'l': 90, 'b': 60, 't': 90, 'r': 30}
        l = dict(orientation="h", x=0, y=-0.3)

    if not (metric2 is None or metric2 == [] or metric2 == metric1): 
        m['r'] = 90

    # Filtern data nach Vergleichszeitraum & Group by zeit
    if vergleichszeitraum is None or vergleichszeitraum == []: # KEIN Vergleichszeitraum 

        if metric2 is None or metric2 == [] or metric2 == metric1: # KEIN Vergleichszeitraum, KEIN KPI2 
            sub_df = data_aktuell[cols + [metric1]]
            traces.append(go.Scatter(x=sub_df[zeit].apply(str), y=sub_df[metric1], mode='lines+markers', name=c+ek))

            lout = go.Layout(xaxis=x, yaxis=y, margin=m, legend=l, hovermode='closest', font=f, showlegend=True)
           
        else: # KEIN Vergleichszeitraum, MIT KPI2 

            sub_df1 = data_aktuell[cols + [metric1]]
            sub_df2 = data_aktuell[cols + [metric2]]
            sub_df = pd.merge(sub_df1, sub_df2, on = cols, how='outer')

            traces.append(go.Scatter(x=sub_df1[zeit].apply(str), y=sub_df1[metric1], mode='lines+markers', name=c+'-'+ek+'-'+metric1))
            traces.append(go.Scatter(x=sub_df2[zeit].apply(str), y=sub_df2[metric2], mode='lines+markers', name=c+'-'+ek+'-'+metric2, yaxis = 'y2'))

            lout = go.Layout(xaxis=x, yaxis=y, yaxis2=y2,margin=m,legend = l,hovermode='closest',font = f, showlegend=True)

    else: #mit Vergleichszeitraum

        zeitlist = ['Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag']

        last_start_date, last_end_date = vergleichsvariable(vergleichszeitraum, start_date, end_date)

        data_vergangen = data_alltime[(data_alltime.Datum >= last_start_date) & (data_alltime.Datum <= last_end_date)]

        data_vergangen = groupslider_filter_data(data_vergangen, zeit)

        format_time_data(data_vergangen, groupslider)

        if metric2 is None or metric2 == [] or metric2 == metric1: # MIT Vergleichszeitraum, KEIN KPI2 

            sub_df1 = data_aktuell[cols + [metric1]]
            sub_df2 = data_vergangen[cols + [metric1]]
            sub_df = sub_df2.append(sub_df1, ignore_index=True)  

            traces.append(go.Scatter(x=sub_df1[zeit].apply(str), y=sub_df1[metric1], mode='lines+markers', name=c+ek+"-aktuell"))
            traces.append(go.Scatter(x=sub_df2[zeit].apply(str), y=sub_df2[metric1], mode='lines+markers', name=c+ek+"-"+vergleichszeitraum, xaxis='x2'))

            lout = go.Layout(xaxis=x,xaxis2=x2,yaxis=y, margin=m,legend = l,hovermode='closest',font = f, showlegend=True)
          
        else: # MIT Vergleichszeitraum, MIT KPI2 
            sub_df1 = data_aktuell[cols + [metric1]]
            sub_df2 = data_vergangen[cols + [metric1]]
            s1 = sub_df2.append(sub_df1, ignore_index=True)    
            sub_df3 = data_aktuell[cols + [metric2]]
            sub_df4 = data_vergangen[cols + [metric2]]
            s2 = sub_df4.append(sub_df3, ignore_index=True)
            sub_df = pd.merge(s1, s2, on = cols, how='outer') 

            traces.append(go.Scatter(x=sub_df1[zeit].apply(str), y=sub_df1[metric1], mode='lines+markers', name=c+ek+'-'+metric1+'-aktuell'))
            traces.append(go.Scatter(x=sub_df2[zeit].apply(str), y=sub_df2[metric1], mode='lines+markers', name=c+ek+'-'+metric1+'-'+vergleichszeitraum, xaxis='x2'))
            traces.append(go.Scatter(x=sub_df3[zeit].apply(str), y=sub_df3[metric2], mode='lines+markers', name=c+ek+'-'+metric2+'-aktuell', yaxis = 'y2'))
            traces.append(go.Scatter(x=sub_df4[zeit].apply(str), y=sub_df4[metric2], mode='lines+markers', name=c+ek+'-'+metric2+'-'+vergleichszeitraum, yaxis = 'y2', xaxis='x2')) 

            lout = go.Layout(xaxis=x, xaxis2=x2, yaxis=y, yaxis2 = y2, margin=m,legend = l, hovermode='closest',font = f, showlegend=True)

    return [sub_df, traces, lout]

def format_time_data(data, groupslider):
    #Format date column
    if groupslider == 1:
        data['Datum'] = pd.to_datetime(data['Datum'], format='%Y-%m-%d')
        data['Datum'] = data['Datum'].dt.strftime('%d/%m/%Y')
    elif groupslider == 2:
        data['Monat'] = pd.to_datetime(data['Monat'], format='%Y-%m')
        data['Monat'] = data['Monat'].dt.strftime('%m/%Y')
    elif groupslider == 3:
        data['Quartal'] = pd.to_datetime(data['Quartal'], format='%Y-%m')
        data['Quartal'] = data['Quartal'].dt.strftime('Q%#m/%Y')
    elif groupslider == 4:
        data['Jahr'] = pd.to_datetime(data['Jahr'], format='%Y')
        data['Jahr'] = data['Jahr'].dt.strftime('%Y')

def format_number_data(data):
    #format last column with metrics depends on column name
    for i in [-2,-1]:
        if(data.columns[i] in ['Nettoumsatz', 'RE1', 'RE2', 'Nettowarenkorb', 'Wareneinsatz']):
            data[data.columns[i]] = data[data.columns[i]].map('{:.2f}€'.format)
        elif(data.columns[i] == 'Marge'):
            data[data.columns[i]] = (data[data.columns[i]]).map('{:.2f}%'.format)

def convert_year_month(someString):
    return [pd.to_datetime(someString, format='%Y-%m').apply(lambda x: x.astype), pd.to_datetime(someString, format='%Y-%m').month]  


##############################################################
#                                                            #
#            I  N  T  E  R  A  C  T  I  O  N  S              #
#                                                            #
##############################################################
#Update columns
@app.callback(
    dash.dependencies.Output('data-table', 'columns'),
    [dash.dependencies.Input('channel1', 'value'),
    dash.dependencies.Input('ekpreis', 'value'),
    dash.dependencies.Input('metric1', 'value'),
    dash.dependencies.Input('metric2', 'value'),
    dash.dependencies.Input('my-date-picker-range', 'start_date'),
    dash.dependencies.Input('my-date-picker-range', 'end_date'),
    dash.dependencies.Input('groupslider', 'value'),
    dash.dependencies.Input('vergleichszeitraum', 'value'),
    ])
def update_columns(channel1, ekpreis, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum):

    if channel1 is None or channel1 == []:
        channel1 = default_shop
    
    if metric1 is None or metric1 == []:
        metric1 = default_metric

    if ekpreis is None or ekpreis == []:
        ekpreis = 'Einkaufspreiskategorie auswählen'
    
    c = channel1[0] 

    if ekpreis == 'Einkaufspreiskategorie auswählen':
        ek = ekpreis
    else:
        ek = ekpreis[0]
    
    result = filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[0]

    cols = list(result.columns)

    new_columns = [{"name": i, "id": i} for i in cols]

    return new_columns
        
#Update table 
@app.callback(
    dash.dependencies.Output('data-table', 'data'),
    [dash.dependencies.Input('channel1', 'value'),
    dash.dependencies.Input('ekpreis', 'value'),
    dash.dependencies.Input('metric1', 'value'),
    dash.dependencies.Input('metric2', 'value'),
    dash.dependencies.Input('my-date-picker-range', 'start_date'),
    dash.dependencies.Input('my-date-picker-range', 'end_date'),
    dash.dependencies.Input('groupslider', 'value'),
    dash.dependencies.Input('vergleichszeitraum', 'value'),
    ])
def update_table(channel1, ekpreis, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum):

    if channel1 is None or channel1 == []:
        channel1 = default_shop
    
    if metric1 is None or metric1 == []:
        metric1 = default_metric

    if ekpreis is None or ekpreis == []:
        ekpreis = 'Einkaufspreiskategorie auswählen'

    sub_df = {}
    data = []

    if ekpreis == 'Einkaufspreiskategorie auswählen':
        ek = ekpreis
        for c in channel1:
            sub_df[c] = filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[0]
            data.append(sub_df[c])
        result = pd.concat(data)
    else: 
        for c in channel1:
            for ek in ekpreis:
                sub_df[c+ek] = filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[0]
                data.append(sub_df[c+ek])
        result = pd.concat(data)
    
    #Format data for show in table
    #format_time_data(result, groupslider)  

    # Save data to csv
    result.to_csv(exportfilename, index=False, encoding='utf-8', float_format='%.2f')

    format_number_data(result)
    
    return result.to_dict('records')
    
#Update graphic
@app.callback(
    dash.dependencies.Output('kpigraph', 'figure'),
    [
        dash.dependencies.Input('channel1', 'value'),
        dash.dependencies.Input('ekpreis', 'value'),
        dash.dependencies.Input('metric1', 'value'),
        dash.dependencies.Input('metric2', 'value'),
        dash.dependencies.Input('my-date-picker-range', 'start_date'),
        dash.dependencies.Input('my-date-picker-range', 'end_date'),
        dash.dependencies.Input('groupslider', 'value'),
        dash.dependencies.Input('vergleichszeitraum', 'value'),
        #dash.dependencies.Input('vergleichbutton', 'n_clicks'),
        dash.dependencies.Input('vergleichcheckbox', 'values'),
    ])

def update_scatterplot(channel1, ekpreis, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum, vergleichcheckbox):

    if channel1 is None or channel1 == []:
        channel1 = default_shop
    
    if metric1 is None or metric1 == []:
        metric1 = default_metric

    if ekpreis is None or ekpreis == []: 
        ekpreis == 'Einkaufspreiskategorie auswählen'

    d = []
    #data = []

    # Diagramme bauen
    if ekpreis == 'Einkaufspreiskategorie auswählen' or ekpreis is None or ekpreis == []:
        ek = 'Einkaufspreiskategorie auswählen'
        for c in channel1:
            #data.append(filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[0])
            d.extend(filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[1])
            l = filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[2]
        #result = pd.concat(data)
    else:
        for c in channel1:
            for ek in ekpreis:
                #data.append(filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[0])
                d.extend(filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[1])
                l = filter_data(c, ek, metric1, metric2, start_date, end_date, groupslider, vergleichszeitraum)[2]
        #result = pd.concat(data)

    if len(d) > 1 and len(channel1) <= 2:
        if vergleichcheckbox == ['abs']:
            l['xaxis2'] = {}
            i = 0
            while i != len(d):
                d[i]['y'] = d[i]['y'] - d[i+1]['y']
                d[i]['name'] = 'Differenz zwischen ' + d[i]['name'] + ' und ' + d[i+1]['name'] 
                del d[i+1]
                i = i + 1
            
        if vergleichcheckbox == ['pro']:
            l['xaxis2'] = {}
            i = 0
            while i != len(d):
                d[i]['y'] = 100 * (d[i]['y'] - d[i+1]['y']) / d[i+1]['y']
                d[i]['name'] = 'Differenz zwischen ' + d[i]['name'] + ' und ' + d[i+1]['name']
                del d[i+1]
                i = i + 1
            
    values = []

    for i in range(0, len(d)):
        values = values + d[i]['y'].tolist()

    scale_axes_values(values, metric1, metric2, l, vergleichszeitraum, vergleichcheckbox)

    # i = 0
    # while i < len(d):
    #     print(d[i]['name'])
    #     i = i + 1

    return {'data': d, 'layout': l}


#Update beirat graphics
@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [
        dash.dependencies.Input('date-picker-beirat', 'start_date'),
        dash.dependencies.Input('date-picker-beirat', 'end_date'),
    ])

def update_beirat_graphen(start_date, end_date):

    zeitlist = ['Datum', 'Jahr', 'Quartal', 'Monat', 'Wochentag']
    zeit = 'Monat'
    zeitlist.remove(zeit)
    metric1 = "Anzahl Bestellungen"

    last_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    last_end_date = last_start_date.replace(day=31).replace(month=12)

    last_start_date = last_start_date.strftime('%Y-%m-%d')
    last_end_date = last_end_date.strftime('%Y-%m-%d')

    data = rohdaten[rohdaten.Kanal.isin(['teufel-shop', 'teufel-ebay'])]

    data_aktuell = data[(data.Datum >= start_date) & (data.Datum <= end_date)]
    data_vergangen = data[(data.Datum >= last_start_date) & (data.Datum <= last_end_date)]

    data_aktuell = data_aktuell.drop(zeitlist, axis = 1).groupby(['Kanal'] + [zeit]).agg(agg_dict).reset_index()
    data_aktuell = cal_marge(data_aktuell)
    format_time_data(data_aktuell, 2)

    data_vergangen = data_vergangen.drop(zeitlist, axis = 1).groupby(['Kanal'] + [zeit]).agg(agg_dict).reset_index()
    data_vergangen = cal_marge(data_vergangen)
    format_time_data(data_vergangen, 2)

    traces = []
    l = {}

    current_month = pd.to_datetime(end_date).month
    current_year = str(pd.to_datetime(end_date).year)
    last_year = str(pd.to_datetime(last_end_date).year)

    for c in ['teufel-shop', 'teufel-ebay']:
        sub_df1 = data_aktuell[data_aktuell.Kanal == c]
        sub_df2 = data_vergangen[data_vergangen.Kanal == c]

        future_months = [word.replace(last_year, current_year) for word in sub_df2[zeit].apply(str).tolist()[current_month:]]

        traces.append(go.Scatter(x=sub_df1[zeit].apply(str).append(pd.Series(future_months, index=range(current_month,12))), 
                                 y=sub_df1[metric1].append(pd.Series(np.repeat(None,len(future_months)), index=range(current_month,12))), mode='lines+markers', name=c+'aktuell'))

        traces.append(go.Scatter(x=sub_df2[zeit].apply(str), y=sub_df2[metric1], mode='lines+markers', name=c+'Vorjahr', xaxis='x2'))

    x2=dict(showgrid = False, type = 'category', overlaying = 'x', side = 'top', titlefont = dict(size = 20))

    ts = ''

    currency_cols = ['Nettoumsatz', 'RE1', 'RE2', 'Nettowarenkorb', 'Wareneinsatz']
    
    # Layout format
    if metric1 in currency_cols:
        ts = "€"
    elif metric1 in ['Marge']:
        ts = "%" 

    m = {'l': 90, 'b': 10, 't': 20, 'r': 30}
    l = dict(orientation="h", x=0, y=-0.2)

    f = dict(family='sans-serif', size=14)

    x=dict(title = '<b>Monat</b>', showgrid = False, titlefont = dict(size = 20), type = 'category')

    y = dict(title = '<b>'+metric1+'</b>', titlefont = dict(size = 20), ticksuffix = ts, tickformat = " .2", zeroline=False, automargin = True, hoverformat = ' .2f')

    l = go.Layout(xaxis=x, yaxis=y, xaxis2 = x2, margin=m, legend=l, hovermode='closest', font=f, showlegend=True)

    return {'data': traces, 'layout': l}


@app.server.route('/downloads/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'downloads'), path
    )

##############################################################
#                                                            #
#                      M  A  I  N                            #
#                                                            #
##############################################################

if __name__ == '__main__':
    #app.run_server(host='127.0.0.1', port=8000, debug=True)
    #app.run_server(host='10.248.209.1')
    app.run_server(debug=True)