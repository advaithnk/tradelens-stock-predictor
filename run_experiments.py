"""
Experimental Results Collection Script for Enigma_24 Conference Paper
This script runs all models and collects performance metrics
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

print("="*80)
print("ENIGMA_24 - EXPERIMENTAL RESULTS COLLECTION")
print("="*80)
print()

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "lstm_results": {},
    "risk_results": {},
    "sentiment_results": {}
}

# ============================================================================
# 1. LSTM PRICE PREDICTION EXPERIMENTS
# ============================================================================
print("\n" + "="*80)
print("1. RUNNING LSTM PRICE PREDICTION EXPERIMENTS")
print("="*80)

try:
    from routes.prediction_analysis import stock_price_predictor
    
    test_stocks = ['AAPL', 'MSFT', 'GOOGL']  # Test with US stocks first
    lstm_metrics = []
    
    for symbol in test_stocks:
        print(f"\nTesting {symbol}...")
        try:
            start_date = datetime(2023, 1, 1)
            end_date = datetime.now()
            
            result = stock_price_predictor(symbol, start_date, end_date)
            
            if 'error' not in result:
                print(f"  ✓ Predicted Price: ${result['predicted_price']}")
                print(f"  ✓ Last Close: ${result['last_close_price']}")
                print(f"  ✓ Change: {result['price_change_percent']}%")
                print(f"  ✓ Direction: {result['prediction_direction']}")
                
                lstm_metrics.append({
                    'symbol': symbol,
                    'predicted_price': result['predicted_price'],
                    'last_close': result['last_close_price'],
                    'change_percent': result['price_change_percent'],
                    'direction': result['prediction_direction']
                })
            else:
                print(f"  ✗ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ✗ Error for {symbol}: {str(e)}")
    
    results['lstm_results'] = {
        'stocks_tested': test_stocks,
        'successful_predictions': len(lstm_metrics),
        'predictions': lstm_metrics
    }
    
    print(f"\n✓ LSTM Experiments Complete: {len(lstm_metrics)}/{len(test_stocks)} successful")
    
except Exception as e:
    print(f"✗ LSTM experiments failed: {str(e)}")
    results['lstm_results'] = {'error': str(e)}

# ============================================================================
# 2. RANDOM FOREST RISK ASSESSMENT EXPERIMENTS
# ============================================================================
print("\n" + "="*80)
print("2. RUNNING RANDOM FOREST RISK ASSESSMENT EXPERIMENTS")
print("="*80)

try:
    from routes.risk_analysis import risk_analysis_model
    
    # Test with Indian stocks
    indian_stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
    risk_metrics = []
    
    for symbol in indian_stocks:
        print(f"\nAnalyzing {symbol}...")
        try:
            result = risk_analysis_model(symbol)
            
            if 'error' not in result:
                print(f"  ✓ Risk Level: {result['risk_level']}")
                print(f"  ✓ Current Price: ₹{result['current_price']}")
                print(f"  ✓ Volatility: {result['volatility']}")
                print(f"  ✓ Trend: {result['trend']}")
                
                risk_metrics.append({
                    'symbol': symbol,
                    'risk_level': result['risk_level'],
                    'current_price': result['current_price'],
                    'volatility': result['volatility'],
                    'trend': result['trend']
                })
            else:
                print(f"  ✗ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ✗ Error for {symbol}: {str(e)}")
    
    results['risk_results'] = {
        'stocks_tested': indian_stocks,
        'successful_assessments': len(risk_metrics),
        'assessments': risk_metrics
    }
    
    print(f"\n✓ Risk Assessment Complete: {len(risk_metrics)}/{len(indian_stocks)} successful")
    
except Exception as e:
    print(f"✗ Risk assessment experiments failed: {str(e)}")
    results['risk_results'] = {'error': str(e)}

# ============================================================================
# 3. SENTIMENT ANALYSIS EXPERIMENTS
# ============================================================================
print("\n" + "="*80)
print("3. RUNNING SENTIMENT ANALYSIS EXPERIMENTS")
print("="*80)

try:
    from routes.sentiment_analysis import fetch_enhanced_news_sentiment, classify_sentiment
    
    test_symbols = ['Apple', 'Microsoft', 'Tesla']
    sentiment_metrics = []
    
    for symbol in test_symbols:
        print(f"\nAnalyzing sentiment for {symbol}...")
        try:
            result = fetch_enhanced_news_sentiment(symbol, num_display=5)
            
            if result['overall_prediction'] is not None:
                print(f"  ✓ Overall Score: {result['overall_prediction']:.2f}")
                print(f"  ✓ Sentiment: {result['overall_sentiment']}")
                print(f"  ✓ Articles Analyzed: {len(result['news'])}")
                
                sentiment_metrics.append({
                    'symbol': symbol,
                    'score': result['overall_prediction'],
                    'sentiment': result['overall_sentiment'],
                    'articles_count': len(result['news'])
                })
            else:
                print(f"  ✗ No sentiment data available")
                
        except Exception as e:
            print(f"  ✗ Error for {symbol}: {str(e)}")
    
    results['sentiment_results'] = {
        'symbols_tested': test_symbols,
        'successful_analyses': len(sentiment_metrics),
        'analyses': sentiment_metrics
    }
    
    print(f"\n✓ Sentiment Analysis Complete: {len(sentiment_metrics)}/{len(test_symbols)} successful")
    
except Exception as e:
    print(f"✗ Sentiment analysis experiments failed: {str(e)}")
    results['sentiment_results'] = {'error': str(e)}

# ============================================================================
# 4. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_file = 'experimental_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENTAL SUMMARY")
print("="*80)

print(f"\n1. LSTM Price Prediction:")
if 'error' not in results['lstm_results']:
    print(f"   - Stocks Tested: {len(results['lstm_results']['stocks_tested'])}")
    print(f"   - Successful: {results['lstm_results']['successful_predictions']}")
else:
    print(f"   - Error: {results['lstm_results']['error']}")

print(f"\n2. Risk Assessment:")
if 'error' not in results['risk_results']:
    print(f"   - Stocks Tested: {len(results['risk_results']['stocks_tested'])}")
    print(f"   - Successful: {results['risk_results']['successful_assessments']}")
else:
    print(f"   - Error: {results['risk_results']['error']}")

print(f"\n3. Sentiment Analysis:")
if 'error' not in results['sentiment_results']:
    print(f"   - Symbols Tested: {len(results['sentiment_results']['symbols_tested'])}")
    print(f"   - Successful: {results['sentiment_results']['successful_analyses']}")
else:
    print(f"   - Error: {results['sentiment_results']['error']}")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE!")
print("="*80)
print(f"\nCheck '{output_file}' for detailed results")
print("Use these results to update your conference paper with actual metrics")
print()
