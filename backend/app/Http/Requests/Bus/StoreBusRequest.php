<?php

namespace App\Http\Requests\Bus;

use Illuminate\Foundation\Http\FormRequest;

class StoreBusRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'plate_number' => 'required|string|max:50|unique:bus,plate_number',
        ];
    }
}
