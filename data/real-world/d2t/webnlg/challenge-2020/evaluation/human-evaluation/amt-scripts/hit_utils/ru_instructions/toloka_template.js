exports.Task = extend(TolokaHandlebarsTask, function(options) {
    TolokaHandlebarsTask.call(this, options);
}, {
    setSolution: function(solution) {
        TolokaHandlebarsTask.prototype.setSolution.apply(this, arguments);
        var workspaceOptions = this.getWorkspaceOptions();

        if (this.rendered) {
            if (!workspaceOptions.isReviewMode && !workspaceOptions.isReadOnly) {
                // Показываем набор чекбоксов, если выбран ответ "Есть нарушения" (BAD), иначе скрываем.
                if (solution.output_values['quality']) {
                    var row = this.getDOMElement().querySelector('.second_scale');
                    row.style.display = solution.output_values['quality'] === 'BAD' ? 'block' : 'none';

                    // Снимаем отметки с чекбоксов, если на первый вопрос выбран вариант ответа "Всё хорошо" (OK).
                    if (solution.output_values['quality'] !== 'BAD') {
                        this.setSolutionOutputValue('advertising', false);
                        this.setSolutionOutputValue('nonsense', false);
                        this.setSolutionOutputValue('insult', false);
                        this.setSolutionOutputValue('law_violation', false);
                        this.setSolutionOutputValue('profanity', false);
                    }
                }
            }
        }
    },

    // Обрабатываем сообщение об ошибке.
    addError: function(message, field, errors) {
        errors || (errors = {
            task_id: this.getOptions().task.id,
            errors: {}
        });
        errors.errors[field] = {
            message: message
        };

        return errors;
    },

    // Проверяем ответы: если выбран ответ "Есть нарушения", нужно отметить хотя бы один чекбокс.
    validate: function(solution) {
        var BAD = solution.output_values['quality'] === 'BAD';
        var errors = null;
        var violationChecked = (solution.output_values.advertising || solution.output_values.nonsense || solution.output_values.insult || solution.output_values.law_violation || solution.output_values.profanity);

        if (BAD && !violationChecked) {
            errors = this.addError("Укажите хотя бы одно нарушение", '__TASK__', errors);
        }

        return errors || TolokaHandlebarsTask.prototype.validate.apply(this, arguments);
    },

    // В режиме проверки раскрываем блок второго вопроса, чтобы увидеть отмеченные исполнителем чекбоксы.
    onRender: function() {
        var workspaceOptions = this.getWorkspaceOptions();

        if (workspaceOptions.isReviewMode || workspaceOptions.isReadOnly || this.getSolution().output_values.quality === 'BAD') {
            var row = this.getDOMElement().querySelector('.second_scale');
            row.style.display = 'block';
        }

        this.rendered = true;
    }
});


exports.Task = extend(TolokaHandlebarsTask, function (options) {
  TolokaHandlebarsTask.call(this, options);
}, {
  onRender: function() {
    // DOM-элемент задания сформирован (доступен через #getDOMElement()) 
        var _document = this.getDOMElement(),
            slider1 = _document.querySelector('.slider1'),
            sliderField1 = _document.querySelector('.slidecontainer1 input[name="coverage"]'),
            slider2 = _document.querySelector('.slider2'),
            sliderField2 = _document.querySelector('.slidecontainer2 input[name="relevance"]'),
            slider3 = _document.querySelector('.slider3'),
            sliderField3 = _document.querySelector('.slidecontainer3 input[name="correctness"]'),     
            slider4 = _document.querySelector('.slider4'),
            sliderField4 = _document.querySelector('.slidecontainer4 input[name="structure"]'),  
            slider5 = _document.querySelector('.slider5'),
            sliderField5 = _document.querySelector('.slidecontainer5 input[name="fluency"]'),         
            task = this;
    
       //выставляем значение поля при изменении ползунка слайдера
          slider1.oninput = function() {
          sliderField1.value = this.value;
          task.setSolutionOutputValue("coverage",this.value);
          };

          slider2.oninput = function() {
          sliderField2.value = this.value;
          task.setSolutionOutputValue("relevance",this.value);
          };

          slider3.oninput = function() {
          sliderField3.value = this.value;
          task.setSolutionOutputValue("correctness",this.value);
          };

          slider4.oninput = function() {
          sliderField4.value = this.value;
          task.setSolutionOutputValue("structure",this.value);
          };  

          slider5.oninput = function() {
          sliderField5.value = this.value;
          task.setSolutionOutputValue("fluency",this.value);
          };            
    },
  
  onDestroy: function() {
    // Задание завершено, можно освобождать (если были использованы) глобальные ресурсы
  }
});




function extend(ParentClass, constructorFunction, prototypeHash) {
  constructorFunction = constructorFunction || function () {};
  prototypeHash = prototypeHash || {};
  if (ParentClass) {
    constructorFunction.prototype = Object.create(ParentClass.prototype);
  }
  for (var i in prototypeHash) {
    constructorFunction.prototype[i] = prototypeHash[i];
  }
  return constructorFunction;
}