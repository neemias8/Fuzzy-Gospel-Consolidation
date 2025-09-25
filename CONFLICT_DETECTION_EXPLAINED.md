# Como o Sistema Detecta Conflitos entre os Evangelhos

## Resposta à Pergunta: "Precisa de dicionários de personagens?"

**SIM**, o sistema melhorado agora usa dicionários estruturados para detectar conflitos de forma mais precisa. Aqui está como funciona:

## 1. **Métodos de Detecção de Conflitos**

### 🔍 **Método 1: Relações Fuzzy (Original)**
- **Como funciona**: Usa similaridade semântica entre textos
- **Detecta**: Textos similares mas não idênticos (0.3 < similaridade < 0.8)
- **Limitações**: Muito restritivo, perdeu conflitos sutis
- **Resultado atual**: Máximo 0.2027 (abaixo do threshold 0.6)

### 🔍 **Método 2: Detecção Melhorada (Novo)**
- **Como funciona**: Usa dicionários estruturados + análise linguística
- **Detecta**: Conflitos específicos por categoria

## 2. **Dicionários Utilizados**

### 👥 **Dicionário de Personagens**
```python
participants = {
    'jesus': {'jesus', 'christ', 'lord', 'master', 'teacher', 'son of man'},
    'peter': {'peter', 'simon', 'simon peter', 'cephas'},
    'judas': {'judas', 'judas iscariot'},
    'pilate': {'pilate', 'pontius pilate', 'governor'},
    # ... mais personagens
}
```

### 📍 **Dicionário de Locais**  
```python
locations = {
    'jerusalem': {'jerusalem', 'holy city'},
    'temple': {'temple', 'temple courts', 'house of god'}, 
    'golgotha': {'golgotha', 'calvary', 'skull', 'place of the skull'},
    # ... mais locais
}
```

### 🔢 **Padrões Numéricos**
```python
numerical_patterns = {
    'rooster_crows': r'(?:rooster|cock).{0,50}(?:crow|crows?)',
    'times_denied': r'(?:deny|disown).{0,30}(?:three|3|twice|two|2)', 
    'pieces_silver': r'(?:thirty|30).{0,20}(?:silver|pieces|coins)'
}
```

### ⏰ **Indicadores Temporais**
```python
temporal_keywords = {
    'early': {'early', 'dawn', 'morning', 'daybreak'},
    'late': {'late', 'evening', 'night', 'dusk'},
    'before': {'before', 'prior to'},
    'after': {'after', 'following', 'then'}
}
```

## 3. **Tipos de Conflitos Detectados**

### 🔴 **Conflitos de Participantes**
- **Exemplo**: "Pedro estava presente" vs. "Pedro não estava lá"
- **Como detecta**: Compara sets de participantes extraídos de cada evento
- **Severidade**: Média

### 🔴 **Conflitos Numéricos** 
- **Exemplo**: "Galo cantou uma vez" vs. "Galo cantou duas vezes"
- **Como detecta**: Regex patterns para números específicos
- **Severidade**: Alta (números são geralmente importantes)

### 🔴 **Conflitos Temporais**
- **Exemplo**: "De manhã cedo" vs. "À noite"  
- **Como detecta**: Indicadores temporais contraditórios
- **Severidade**: Média

### 🔴 **Conflitos de Local**
- **Exemplo**: "No templo" vs. "Na sinagoga"
- **Como detecta**: Locais diferentes para mesmo evento
- **Severidade**: Alta (local é crucial)

### 🔴 **Conflitos de Sequência**
- **Exemplo**: Ordem diferente de eventos no mesmo dia
- **Como detecta**: Análise da sequência temporal
- **Severidade**: Média

## 4. **Casos de Teste Conhecidos (Config.yaml)**

### 📋 **Casos Configurados**:

1. **Pedro negando Jesus** (eventos 67, 89, 91)
   - **Conflito**: "Número de vezes que o galo canta"
   - **Detecção**: Regex pattern + contagem numérica

2. **Limpeza do templo** (eventos 12, 45)
   - **Conflito**: "Timing da limpeza do templo" 
   - **Detecção**: Análise temporal + localização

3. **Entrada triunfal** (eventos 4, 5, 6)
   - **Conflito**: "Detalhes do animal"
   - **Detecção**: Dicionário de objetos/animais

## 5. **Como Funciona na Prática**

### 🔄 **Processo de Detecção**:

1. **Pré-processamento**:
   - Extrai texto de todos os evangelhos para cada evento
   - Normaliza e limpa o texto

2. **Análise por Categoria**:
   - Para cada par de eventos relacionados:
     - Extrai participantes usando dicionário
     - Busca padrões numéricos com regex
     - Identifica indicadores temporais
     - Mapeia localizações mencionadas

3. **Comparação e Conflito**:
   - Compara extrações entre eventos
   - Identifica discrepâncias
   - Calcula score de severidade

4. **Resolução no Resumo**:
   - Procura indicadores de resolução:
     - "According to Matthew..."
     - "Alternative accounts..."
     - "Some gospels report..."

## 6. **Vantagens do Método Melhorado**

### ✅ **Precisão**:
- Detecta conflitos específicos conhecidos
- Usa conhecimento teológico estruturado

### ✅ **Flexibilidade**: 
- Fácil adicionar novos tipos de conflito
- Dicionários expansíveis

### ✅ **Interpretabilidade**:
- Explica QUAL tipo de conflito
- Mostra evidência textual específica

## 7. **Implementação Atual vs. Ideal**

### 🎯 **Status Atual**:
- ✅ Estrutura implementada
- ✅ Dicionários básicos criados  
- ✅ Métodos de detecção definidos
- ⚠️ Precisa integrar com dados reais do corpus

### 🎯 **Próximos Passos**:
1. Integrar detector melhorado com corpus real
2. Ajustar thresholds baseado em dados
3. Expandir dicionários com mais variantes
4. Implementar análise de resolução de conflitos

## 8. **Exemplo de Saída**

```json
{
  "conflict_type": "numerical_conflict",
  "pattern": "rooster_crows", 
  "event1_values": ["cock crowed once"],
  "event2_values": ["rooster crowed twice"],
  "severity": "high",
  "gospels": ["mark", "matthew"]
}
```

**Conclusão**: O sistema agora tem capacidades muito mais sofisticadas de detecção de conflitos, usando conhecimento estruturado sobre personagens, locais, números e tempo para identificar discrepâncias específicas entre os relatos dos Evangelhos.