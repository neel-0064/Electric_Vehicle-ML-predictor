from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import base64
import io
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle

app = Dash(__name__, suppress_callback_exceptions=True)

#For deployment on GCP using gunicorn
server = app.server

app.layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Upload File']),
        style={
            'width': '100%', 'height': '50px', 'lineHeight': '50px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '10px'
        }
    ),
    html.Div(id='upload-output'),
    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-model'),
    dcc.Store(id='stored-features'),

    html.Div([
        html.Div([
            html.Label("Select Target:"),
            dcc.Dropdown(id='target-dropdown', style={'width': '200px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'justifyContent': 'center'}),
    ], id='target-section', style={'display': 'none', 'marginBottom': '10px'}),

    html.Div(id='target-summary'),

    html.Div([
        html.Div(id='avg-chart', style={'flex': 1, 'minWidth': 0}),
        html.Div(id='corr-chart', style={'flex': 1, 'minWidth': 0}),
    ], style={'display': 'flex', 'gap': '20px'}),

    html.Div([
        html.Label("Select Features:", style={'display': 'block', 'textAlign': 'center'}),
        dcc.Checklist(id='feature-checklist', inline=True,
                      style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'}),
        html.Br(),
        html.Div(html.Button('Train', id='train-btn', n_clicks=0),
                 style={'textAlign': 'center'}),
        html.Div(id='train-output', style={'textAlign': 'center'})
    ], id='train-section', style={'display': 'none', 'marginBottom': '10px'}),

    html.Div(id='predict-section')

], style={'maxWidth': '900px', 'margin': '0 auto', 'padding': '20px'})


@app.callback(
    Output('upload-output', 'children'),
    Output('stored-data', 'data'),
    Output('target-dropdown', 'options'),
    Output('target-dropdown', 'value'),
    Output('target-section', 'style'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def upload_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    expected_cols = ['County', 'City', 'Make', 'Model', 'Legislative District']
    
    if all(col in df.columns for col in expected_cols):
        df = df.dropna(subset=expected_cols)
    #We check if the columns exist in case another data file is provided
    if 'Electric Range' in df.columns and 'Model Year' in df.columns:
        df = df[(df['Electric Range'] > 0) & (df['Model Year'] <= 2025)]

    #This is a fix for the size column being a string in the tips database. Although this code was developed based on the electric vehicle database, we added this in case it is tested on the tips database.
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9, 'ten': 10
    }
    if 'size' in df.columns:
        df['size'] = df['size'].astype(str).str.lower().map(word_to_num).fillna(df['size'])
        df['size'] = pd.to_numeric(df['size'], errors='coerce')

    num_cols = df.select_dtypes(include='number').columns.tolist()
    options = [{'label': col, 'value': col} for col in num_cols]
    default = 'Electric Range' if 'Electric Range' in num_cols else num_cols[0]
    return html.P(""), df.to_json(orient='split'), options, default, {'display': 'block'}


def make_avg_chart(df, default_cat, target):
    if pd.api.types.is_numeric_dtype(df[target]):
        avg = df.groupby(default_cat)[target].mean().reset_index()
        y_col = target
        y_label = f'{target} (average)'
    else:
        avg = df.groupby(default_cat)[target].count().reset_index()
        avg.columns = [default_cat, 'Count']
        y_col = 'Count'
        y_label = 'Count'
    fig = px.bar(avg, x=default_cat, y=y_col,
                 title=f'Average {target} by {default_cat}',
                 labels={y_col: y_label})
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=100), title_x=0.5,
                      xaxis_tickangle=-45)
    return dcc.Graph(figure=fig, style={'height': '350px', 'width': '100%'}, config={'responsive': True})


@app.callback(
    Output('target-summary', 'children'),
    Output('feature-checklist', 'options'),
    Output('feature-checklist', 'value'),
    Output('train-section', 'style'),
    Output('avg-chart', 'children'),
    Output('corr-chart', 'children'),
    Input('target-dropdown', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def on_target(target, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    summary = html.P("")

    feat_options = [{'label': c, 'value': c} for c in df.columns if c != target]
    default_feats = [f for f in ['Model Year', 'Base MSRP', 'Make', 'Electric Vehicle Type'] if f in df.columns and f != target]

    cat_cols = [c for c in df.select_dtypes(exclude='number').columns.tolist() if c != target]
    default_cat = cat_cols[0] if cat_cols else None

    if default_cat:
        initial_chart = make_avg_chart(df, default_cat, target)
    else:
        initial_chart = html.P("No categorical features available.")

    avg_chart = html.Div([
        dcc.RadioItems(
            id='cat-radio',
            options=[{'label': c, 'value': c} for c in cat_cols],
            value=default_cat,
            inline=True
        ),
        html.Div(initial_chart, id='avg-chart-inner')
    ])

    num_feats = [c for c in df.select_dtypes(include='number').columns.tolist() if c != target]
    y = pd.to_numeric(df[target], errors='coerce')
    corr = pd.Series([df[f].corr(y) for f in num_feats], index=num_feats).abs().sort_values()
    fig = px.bar(x=corr.index, y=corr.values,
                 title='Correlation Strength of Numerical Variables with ' + target,
                 labels={'x': 'Numerical Variables', 'y': 'Correlation Strength (Absolute Value)'})
    fig.update_layout(
        margin=dict(l=100, r=30, t=80, b=100),
        title_x=0.5,
        title_font_size=12,
        xaxis_tickangle=-45
    )
    corr_chart = dcc.Graph(figure=fig, style={'height': '400px', 'width': '100%'}, config={'responsive': True})

    return summary, feat_options, default_feats, {'display': 'block', 'marginBottom': '10px'}, avg_chart, corr_chart


@app.callback(
    Output('avg-chart-inner', 'children'),
    Input('cat-radio', 'value'),
    State('target-dropdown', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def show_avg_chart(cat_col, target, data):
    if not cat_col:
        return html.P("No categorical features available.")
    df = pd.read_json(io.StringIO(data), orient='split')
    return make_avg_chart(df, cat_col, target)


@app.callback(
    Output('train-output', 'children'),
    Output('stored-model', 'data'),
    Output('stored-features', 'data'),
    Output('predict-section', 'children'),
    Input('train-btn', 'n_clicks'),
    State('feature-checklist', 'value'),
    State('target-dropdown', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def train(n_clicks, features, target, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    X = df[features].copy()
    y = pd.to_numeric(df[target], errors='coerce')
    mask = y.notna()
    X, y = X[mask], y[mask]

    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                              ('scl', StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                              ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                                     sparse_output=False))]), cat_cols))
    #creating the full pipeling with encoding and random forest regressor
    pipe = Pipeline([('pre', ColumnTransformer(transformers)), ('model', RandomForestRegressor(random_state=42))])
    
    
    param_grid = {
        'model__n_estimators': [10, 50], 
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
    gs.fit(X_train, y_train)

    best_pipe = gs.best_estimator_
    r2 = r2_score(y_test, best_pipe.predict(X_test))




    model_b64 = base64.b64encode(pickle.dumps({
        'pipe': best_pipe, 'features': features, 'num_cols': num_cols, 'target': target
    })).decode()

    train_output = html.P(f"The R² score is: {r2:.4f}")

    placeholder = ', '.join(features)
    predict_section = html.Div([
        html.Div([
            dcc.Input(
                id='pred-input-text',
                type='text',
                placeholder=f'Enter values: {placeholder}',
                style={'width': '400px', 'marginRight': '10px', 'padding': '5px'}
            ),
            html.Button('Predict', id='pred-btn', n_clicks=0),
            html.Span(id='pred-output', style={'marginLeft': '15px', 'fontWeight': 'bold'})
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '5px'})
    ])

    return train_output, model_b64, features, predict_section


@app.callback(
    Output('pred-output', 'children'),
    Input('pred-btn', 'n_clicks'),
    State('pred-input-text', 'value'),
    State('stored-model', 'data'),
    prevent_initial_call=True
)
def predict(n, input_text, model_b64):
    if not input_text:
        return "Please enter values."

    obj = pickle.loads(base64.b64decode(model_b64))
    pipe, features, num_cols, target = obj['pipe'], obj['features'], obj['num_cols'], obj['target']

    values = [v.strip() for v in input_text.split(',')]
    if len(values) != len(features):
        return f"Expected {len(features)} values, got {len(values)}."

    row = {}
    for feat, val in zip(features, values):
        if feat in num_cols:
            try:
                row[feat] = float(val)
            except ValueError:
                return f"Invalid number for {feat}: {val}"
        else:
            row[feat] = val

    try:
        pred = pipe.predict(pd.DataFrame([row]))[0]
        return f"Predicted {target} is: {pred:.2f}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=False)