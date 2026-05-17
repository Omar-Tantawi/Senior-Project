<?php

namespace App\Http\Requests\Section;

use Illuminate\Foundation\Http\FormRequest;

class UpdateSectionRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'class_id' => 'sometimes|exists:class,class_id',
            'name'     => 'sometimes|string|max:100',
        ];
    }
}
