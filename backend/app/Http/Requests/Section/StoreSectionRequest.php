<?php

namespace App\Http\Requests\Section;

use Illuminate\Foundation\Http\FormRequest;

class StoreSectionRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'class_id' => 'required|exists:class,class_id',
            'name'     => 'required|string|max:100',
        ];
    }
}
