from gradient_boosting_model import HeartDiseaseEnsembleModel

if __name__ == '__main__':
    model = HeartDiseaseEnsembleModel()
    model.load_and_comprehensive_analysis('heart.csv')
    model.advanced_feature_engineering()
    model.build_and_evaluate_models()
    model.create_ensemble_and_evaluate()
    model.display_results()
    model.save_model()
    print("\n✅ End-to-end pipeline executed successfully!")